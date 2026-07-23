from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"

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
from qwen_omd_dataloaders.config import ModuleCConfig  # noqa: E402
from qwen_omd_dataloaders.datasets import EgoOopsModuleCDataset, EgoOopsProvider  # noqa: E402
from qwen_omd_dataloaders.schema import TrainingExample  # noqa: E402

from train_module_a_unsloth import (  # noqa: E402
    EpochCheckpointCallback,
    Qwen3VLMetadataVisionDataCollator,
    grouped_video_ids,
    is_bf16_available,
    json_safe,
    load_generation_eval_model,
    load_unsloth_model,
    release_cuda_memory,
    save_outputs,
    validate_lora_target_configuration,
    validate_video_split,
)

MODULE_C_SPLIT_VERSION = "module_c_video_split_v1"
MODULE_C_SYSTEM_PROMPT = (
    "You are a video mistake-detection and reasoning model. Compare the visible execution in the "
    "provided bounded clip to the written instruction. Decide whether there is a procedural mistake, "
    "then explain the visual evidence briefly. Return strict compact JSON only, with exactly these keys: "
    "{\"mistake\":true|false,\"reasoning\":\"short explanation\"}. Do not include markdown or extra text."
)


def disable_unsloth_extra_outputs() -> None:
    was_logits = os.environ.get("UNSLOTH_RETURN_LOGITS")
    was_hidden = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    if was_logits == "1" or was_hidden == "1":
        print(
            "Module C disabled leaked Unsloth extra outputs: "
            f"UNSLOTH_RETURN_LOGITS={was_logits}, "
            f"UNSLOTH_RETURN_HIDDEN_STATES={was_hidden}"
        )
    os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen VL for Module C mistake reasoning.")
    parser.add_argument("--metadata", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--mistake-classes", default=DEFAULT_MISTAKE_CLASSES_PATH)
    parser.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--video-ids", nargs="*", default=None)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--max-videos", type=int, default=50)
    parser.add_argument("--model-name", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-16bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune-vision-layers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--finetune-language-layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune-attention-modules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune-mlp-modules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-modules",
        choices=("auto", "all-linear"),
        default="auto",
        help=(
            "LoRA target selection. 'auto' respects finetune_* flags. "
            "'all-linear' targets every linear layer and requires all finetune_* flags to be true."
        ),
    )
    parser.add_argument("--train-mode", choices=["steps", "epochs"], default="steps")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--checkpoint-epochs", type=int, default=1)
    parser.add_argument("--eval-epochs", type=int, default=1)
    parser.add_argument("--keep-last-checkpoints", type=int, default=4)
    parser.add_argument("--keep-best-checkpoints", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--metric-for-best-model", default="eval_mistake/f1")
    parser.add_argument("--greater-is-better", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--val-videos-per-task", type=int, default=2)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--regenerate-split", action="store_true")
    parser.add_argument("--module-c-min-duration-seconds", type=float, default=0.5)
    parser.add_argument("--module-c-jitter-ratio-min", type=float, default=0.05)
    parser.add_argument("--module-c-jitter-ratio-max", type=float, default=0.10)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--eval-max-frames", type=int, default=None)
    parser.add_argument("--vision-resize", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--eval-generation-max-samples", type=int, default=-1)
    parser.add_argument("--eval-generation-max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--generation-eval-only",
        action="store_true",
        help="Skip training and run Module C generation metrics on the validation split only.",
    )
    parser.add_argument(
        "--eval-checkpoint",
        default=None,
        help="LoRA checkpoint directory for --generation-eval-only.",
    )
    parser.add_argument(
        "--generation-eval-output-json",
        default=None,
        help="Optional path to write generation-eval-only metrics JSON.",
    )
    parser.add_argument("--output-dir", default="outputs/module_c_qwen35_lora_reasoning")
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-examples", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--wandb-log-best-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-artifact-prefix", default="module-c")
    args = parser.parse_args()
    args.report_to = [item.strip() for item in args.report_to.split(",") if item.strip()]
    if args.max_videos <= 0:
        parser.error("--max-videos must be positive.")
    if args.eval_max_frames is None:
        args.eval_max_frames = args.max_frames
    if args.max_frames < args.min_frames or args.eval_max_frames < args.min_frames:
        parser.error("--max-frames and --eval-max-frames must be >= --min-frames.")
    if args.module_c_min_duration_seconds <= 0:
        parser.error("--module-c-min-duration-seconds must be positive.")
    if args.module_c_jitter_ratio_min < 0 or args.module_c_jitter_ratio_max < args.module_c_jitter_ratio_min:
        parser.error("Module C jitter ratios must be non-negative and min <= max.")
    if args.keep_last_checkpoints < 0 or args.keep_best_checkpoints < 0:
        parser.error("checkpoint retention counts cannot be negative.")
    if args.eval_generation_max_samples < -1:
        parser.error("--eval-generation-max-samples must be -1, 0, or positive.")
    if args.generation_eval_only and not args.eval_checkpoint:
        parser.error("--generation-eval-only requires --eval-checkpoint.")
    if args.generation_eval_only and args.dry_run:
        parser.error("--generation-eval-only cannot be combined with --dry-run.")
    validate_lora_target_configuration(args, parser.error)
    return args


def build_module_c_dataset(args: argparse.Namespace) -> EgoOopsModuleCDataset:
    provider = EgoOopsProvider(
        metadata_path=args.metadata,
        mistake_classes_path=args.mistake_classes,
        video_root=args.video_root,
        video_ids=set(args.video_ids) if args.video_ids else None,
        task_ids=set(args.task_ids) if args.task_ids else None,
        max_videos=args.max_videos,
        require_existing_videos=True,
    )
    config = ModuleCConfig(
        min_duration_seconds=args.module_c_min_duration_seconds,
        jitter_ratio_min=args.module_c_jitter_ratio_min,
        jitter_ratio_max=args.module_c_jitter_ratio_max,
        seed=args.seed,
    )
    return EgoOopsModuleCDataset(provider, config=config)


def default_split_file(args: argparse.Namespace) -> Path:
    return PROJECT_ROOT / "splits" / f"module_c_video_split_seed{args.seed}.json"


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
        "module_c_min_duration_seconds": args.module_c_min_duration_seconds,
        "module_c_jitter_ratio_min": args.module_c_jitter_ratio_min,
        "module_c_jitter_ratio_max": args.module_c_jitter_ratio_max,
    }


def make_module_c_video_split(dataset: EgoOopsModuleCDataset, args: argparse.Namespace) -> dict[str, Any]:
    grouped = grouped_video_ids(dataset)
    if not grouped:
        raise ValueError("Cannot create a split from an empty dataset.")

    import random

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
        "version": MODULE_C_SPLIT_VERSION,
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


def load_or_create_video_split(dataset: EgoOopsModuleCDataset, args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    split_path = Path(args.split_file) if args.split_file else default_split_file(args)
    if split_path.exists() and not args.regenerate_split:
        with open(split_path, "r", encoding="utf-8") as file:
            split = json.load(file)
        validate_video_split(split, dataset)
        return split, split_path
    split = make_module_c_video_split(dataset, args)
    validate_video_split(split, dataset)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as file:
        json.dump(split, file, indent=2, sort_keys=True)
        file.write("\n")
    return split, split_path


def training_example_to_conversation(
    example: TrainingExample,
    *,
    fps: float,
    min_frames: int,
    max_frames: int,
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": MODULE_C_SYSTEM_PROMPT}]},
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
                    {"type": "text", "text": example.prompt_text},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": example.target_text}]},
        ],
        "video_id": example.video_id,
        "task_id": example.task_id,
        "step_index": example.step_index,
        "window_start": example.window_start,
        "window_end": example.window_end,
        "gt_start": example.gt_start,
        "gt_end": example.gt_end,
        "target": example.target_text,
        "label": example.label,
        "mistake": example.mistake,
        "reasoning": example.reasoning,
        "metadata": example.metadata,
    }


def conversations_for_video_ids(
    dataset: EgoOopsModuleCDataset,
    video_ids: set[str],
    *,
    fps: float,
    min_frames: int,
    max_frames: int,
) -> list[dict[str, Any]]:
    return [
        training_example_to_conversation(example, fps=fps, min_frames=min_frames, max_frames=max_frames)
        for example in dataset
        if example.video_id in video_ids
    ]


def summarize_dataset(dataset: EgoOopsModuleCDataset) -> dict[str, Any]:
    labels = Counter(example.label for example in dataset)
    return {
        "num_examples": len(dataset),
        "num_videos": len({record.video_id for record in dataset.records}),
        "label_counts": dict(sorted(labels.items())),
        "tasks": {
            task_id: len(video_ids)
            for task_id, video_ids in grouped_video_ids(dataset).items()
        },
    }


def summarize_conversations(conversations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_examples": len(conversations),
        "num_videos": len({item["video_id"] for item in conversations}),
        "label_counts": dict(sorted(Counter(str(item.get("label")) for item in conversations).items())),
    }


def save_run_config(
    *,
    args: argparse.Namespace,
    dataset: EgoOopsModuleCDataset,
    train_conversations: list[dict[str, Any]],
    val_conversations: list[dict[str, Any]],
    split: dict[str, Any],
    split_path: Path,
) -> None:
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": json_safe(vars(args)),
        "dataset_summary": summarize_dataset(dataset),
        "num_conversations": len(train_conversations) + len(val_conversations),
        "split_file": str(split_path),
        "split": json_safe(split),
        "train_summary": summarize_conversations(train_conversations),
        "validation_summary": summarize_conversations(val_conversations),
    }
    config_path = output_path / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")
    print(f"Saved run config to: {config_path}")


def parse_bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "mistake"}:
            return True
        if normalized in {"false", "no", "0", "correct"}:
            return False
    return None


def parse_module_c_prediction(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    mistake = parse_bool_value(payload.get("mistake"))
    reasoning = payload.get("reasoning")
    if mistake is None or not isinstance(reasoning, str) or not reasoning.strip():
        return None
    return {"mistake": mistake, "reasoning": reasoning.strip()}


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def lexical_overlap(reference: str, prediction: str) -> float:
    reference_tokens = _token_set(reference)
    prediction_tokens = _token_set(prediction)
    if not reference_tokens and not prediction_tokens:
        return 1.0
    if not reference_tokens or not prediction_tokens:
        return 0.0
    return len(reference_tokens & prediction_tokens) / len(reference_tokens | prediction_tokens)


def module_c_metrics(
    targets: list[bool],
    predictions: list[bool | None],
    *,
    target_reasonings: list[str] | None = None,
    predicted_reasonings: list[str | None] | None = None,
) -> dict[str, float]:
    tp = fp = tn = fn = invalid = 0
    for target, prediction in zip(targets, predictions):
        if prediction is None:
            invalid += 1
            if target:
                fn += 1
            else:
                fp += 1
        elif target and prediction:
            tp += 1
        elif not target and prediction:
            fp += 1
        elif not target and not prediction:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0
    metrics = {
        "eval_mistake/accuracy": accuracy,
        "eval_mistake/precision": precision,
        "eval_mistake/recall": recall,
        "eval_mistake/f1": f1,
        "eval_mistake/specificity": specificity,
        "eval_mistake/balanced_accuracy": (recall + specificity) / 2.0,
        "eval_mistake/mcc": mcc,
        "eval_mistake/false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "eval_mistake/false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "eval_mistake/invalid_json_rate": invalid / total if total else 0.0,
        "eval_mistake/tp": float(tp),
        "eval_mistake/fp": float(fp),
        "eval_mistake/tn": float(tn),
        "eval_mistake/fn": float(fn),
        "eval_mistake/num_samples": float(total),
    }
    if target_reasonings is not None and predicted_reasonings is not None:
        valid_reasonings = [
            (target, prediction)
            for target, prediction in zip(target_reasonings, predicted_reasonings)
            if prediction is not None and prediction.strip()
        ]
        metrics["eval_reasoning/nonempty_rate"] = len(valid_reasonings) / total if total else 0.0
        metrics["eval_reasoning/lexical_overlap"] = (
            sum(lexical_overlap(target, prediction) for target, prediction in valid_reasonings) / len(valid_reasonings)
            if valid_reasonings else 0.0
        )
    return metrics


class ModuleCMistakeMetricsCallback:
    def __init__(
        self,
        *,
        eval_examples: list[dict[str, Any]],
        data_collator: Qwen3VLMetadataVisionDataCollator,
        tokenizer: Any,
        max_samples: int,
        max_new_tokens: int,
    ) -> None:
        from transformers import TrainerCallback

        self._callback_base = TrainerCallback()
        self.trainer: Any | None = None
        self.eval_examples = eval_examples if max_samples < 0 else eval_examples[:max_samples]
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def __getattr__(self, name: str) -> Any:
        return getattr(self._callback_base, name)

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.trainer is None or not self.eval_examples:
            return control
        model = kwargs["model"]
        was_training = model.training
        predictions: list[bool | None] = []
        targets: list[bool] = []
        predicted_reasonings: list[str | None] = []
        target_reasonings: list[str] = []

        import torch
        from unsloth import FastVisionModel

        FastVisionModel.for_inference(model)
        try:
            with torch.inference_mode():
                for example in self.eval_examples:
                    generated_text = self._generate_json(model, example)
                    parsed = parse_module_c_prediction(generated_text)
                    predictions.append(None if parsed is None else bool(parsed["mistake"]))
                    predicted_reasonings.append(None if parsed is None else str(parsed["reasoning"]))
                    targets.append(bool(example.get("mistake")))
                    target_reasonings.append(str(example.get("reasoning") or ""))
        finally:
            if was_training:
                FastVisionModel.for_training(model)
            disable_unsloth_extra_outputs()
            release_cuda_memory()

        metrics = module_c_metrics(
            targets,
            predictions,
            target_reasonings=target_reasonings,
            predicted_reasonings=predicted_reasonings,
        )
        callback_metrics = kwargs.get("metrics")
        if isinstance(callback_metrics, dict):
            callback_metrics.update(metrics)
        self.trainer.log(metrics)
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

    def _generate_json(self, model: Any, example: dict[str, Any]) -> str:
        inputs = self._generation_inputs(example)
        device = next(model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]
        generate_kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": False, "use_cache": False}
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


class ModuleCUnslothEnvGuardCallback:
    def __init__(self) -> None:
        from transformers import TrainerCallback

        self._callback_base = TrainerCallback()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._callback_base, name)

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        disable_unsloth_extra_outputs()
        return control

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        disable_unsloth_extra_outputs()
        return control

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        disable_unsloth_extra_outputs()
        return control


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
    )
    mistake_callback = ModuleCMistakeMetricsCallback(
        eval_examples=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        max_samples=args.eval_generation_max_samples,
        max_new_tokens=args.eval_generation_max_new_tokens,
    )
    callbacks.append(mistake_callback)
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
        generation_eval_args=argparse.Namespace(
            eval_generation_max_samples=0,
            generation_eval_mode="inline",
        ),
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(ModuleCUnslothEnvGuardCallback())
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_eval_dataset,
        data_collator=data_collator,
        args=SFTConfig(**trainer_args),
        callbacks=callbacks,
    )
    mistake_callback.trainer = trainer
    checkpoint_callback.trainer = trainer
    return trainer


def run_generation_eval_only(
    *,
    args: argparse.Namespace,
    val_conversations: list[dict[str, Any]],
) -> None:
    import torch
    from unsloth import FastVisionModel

    if args.eval_generation_max_samples == 0:
        metrics: dict[str, float] = {}
        predictions: list[dict[str, Any]] = []
    else:
        model, tokenizer = load_generation_eval_model(args)
        collator = Qwen3VLMetadataVisionDataCollator(
            model,
            tokenizer,
            max_seq_length=args.max_seq_length,
            resize=args.vision_resize,
            train_on_responses_only=True,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
            completion_only_loss=True,
        )
        generator = ModuleCMistakeMetricsCallback(
            eval_examples=val_conversations,
            data_collator=collator,
            tokenizer=tokenizer,
            max_samples=args.eval_generation_max_samples,
            max_new_tokens=args.eval_generation_max_new_tokens,
        )
        predictions = []
        pred_labels: list[bool | None] = []
        target_labels: list[bool] = []
        predicted_reasonings: list[str | None] = []
        target_reasonings: list[str] = []
        FastVisionModel.for_inference(model)
        try:
            with torch.inference_mode():
                for example in generator.eval_examples:
                    generated_text = generator._generate_json(model, example)
                    parsed = parse_module_c_prediction(generated_text)
                    pred_mistake = None if parsed is None else bool(parsed["mistake"])
                    pred_reasoning = None if parsed is None else str(parsed["reasoning"])
                    pred_labels.append(pred_mistake)
                    predicted_reasonings.append(pred_reasoning)
                    target_labels.append(bool(example.get("mistake")))
                    target_reasonings.append(str(example.get("reasoning") or ""))
                    predictions.append(
                        {
                            "video_id": example.get("video_id"),
                            "task_id": example.get("task_id"),
                            "step_index": example.get("step_index"),
                            "window_start": example.get("window_start"),
                            "window_end": example.get("window_end"),
                            "gt_mistake": bool(example.get("mistake")),
                            "gt_reasoning": example.get("reasoning"),
                            "pred_mistake": pred_mistake,
                            "pred_reasoning": pred_reasoning,
                            "raw": generated_text,
                        }
                    )
        finally:
            del model
            disable_unsloth_extra_outputs()
            release_cuda_memory()
        metrics = module_c_metrics(
            target_labels,
            pred_labels,
            target_reasonings=target_reasonings,
            predicted_reasonings=predicted_reasonings,
        )

    payload = {
        "checkpoint": args.eval_checkpoint,
        "num_validation_examples": len(val_conversations),
        "num_generation_examples": (
            len(val_conversations)
            if args.eval_generation_max_samples < 0
            else min(args.eval_generation_max_samples, len(val_conversations))
        ),
        "eval_max_frames": args.eval_max_frames,
        "vision_resize": args.vision_resize,
        "max_seq_length": args.max_seq_length,
        "metrics": metrics,
        "predictions": predictions,
    }
    if args.generation_eval_output_json:
        output_path = Path(args.generation_eval_output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(json_safe(payload), file, indent=2, sort_keys=True, ensure_ascii=False)
            file.write("\n")
        print(f"Saved Module C generation eval to: {output_path}")
    print(json.dumps(json_safe({key: value for key, value in payload.items() if key != "predictions"}), indent=2, sort_keys=True))


def print_dry_run(
    dataset: EgoOopsModuleCDataset,
    train_conversations: list[dict[str, Any]],
    val_conversations: list[dict[str, Any]],
    split: dict[str, Any],
    split_path: Path,
    limit: int,
) -> None:
    print("Module C full dataset summary:")
    print(json.dumps(summarize_dataset(dataset), indent=2, ensure_ascii=False))
    print(f"Module C split file: {split_path}")
    print("Module C split video summary:")
    print(json.dumps({
        "train_video_ids": split["train_video_ids"],
        "val_video_ids": split["val_video_ids"],
        "tasks": split["tasks"],
    }, indent=2, ensure_ascii=False))
    print("Module C train summary:")
    print(json.dumps(summarize_conversations(train_conversations), indent=2, ensure_ascii=False))
    print("Module C validation summary:")
    print(json.dumps(summarize_conversations(val_conversations), indent=2, ensure_ascii=False))
    for index, conversation in enumerate(train_conversations[: max(0, limit)]):
        print(f"\nExample {index}:")
        print(json.dumps(conversation, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    print(
        "Module C Unsloth env: "
        f"UNSLOTH_RETURN_LOGITS={os.environ.get('UNSLOTH_RETURN_LOGITS')}, "
        f"UNSLOTH_RETURN_HIDDEN_STATES={os.environ.get('UNSLOTH_RETURN_HIDDEN_STATES')}"
    )
    dataset = build_module_c_dataset(args)
    split, split_path = load_or_create_video_split(dataset, args)
    train_conversations = conversations_for_video_ids(
        dataset,
        set(split["train_video_ids"]),
        fps=args.fps,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
    )
    val_conversations = conversations_for_video_ids(
        dataset,
        set(split["val_video_ids"]),
        fps=args.fps,
        min_frames=args.min_frames,
        max_frames=args.eval_max_frames,
    )
    if not train_conversations or not val_conversations:
        raise ValueError("Train/validation split produced an empty conversation split.")
    if args.generation_eval_only:
        print("Module C full dataset summary:")
        print(json.dumps(summarize_dataset(dataset), indent=2, ensure_ascii=False))
        print(f"Module C split file: {split_path}")
        print("Module C validation summary:")
        print(json.dumps(summarize_conversations(val_conversations), indent=2, ensure_ascii=False))
        run_generation_eval_only(args=args, val_conversations=val_conversations)
        return
    if args.dry_run:
        print_dry_run(dataset, train_conversations, val_conversations, split, split_path, args.print_examples)
        return
    print("Module C full dataset summary:")
    print(json.dumps(summarize_dataset(dataset), indent=2, ensure_ascii=False))
    print(f"Module C split file: {split_path}")
    print("Module C train summary:")
    print(json.dumps(summarize_conversations(train_conversations), indent=2, ensure_ascii=False))
    print("Module C validation summary:")
    print(json.dumps(summarize_conversations(val_conversations), indent=2, ensure_ascii=False))

    save_run_config(
        args=args,
        dataset=dataset,
        train_conversations=train_conversations,
        val_conversations=val_conversations,
        split=split,
        split_path=split_path,
    )
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
