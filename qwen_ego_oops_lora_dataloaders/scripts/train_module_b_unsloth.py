from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
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
from qwen_omd_dataloaders.config import ModuleAConfig, ModuleBConfig  # noqa: E402
from qwen_omd_dataloaders.datasets import EgoOopsModuleBDataset, EgoOopsProvider  # noqa: E402
from qwen_omd_dataloaders.schema import TrainingExample  # noqa: E402

from train_module_a_unsloth import (  # noqa: E402
    EpochCheckpointCallback,
    Qwen3VLMetadataVisionDataCollator,
    grouped_video_ids,
    is_bf16_available,
    json_safe,
    load_unsloth_model,
    make_video_split,
    release_cuda_memory,
    save_outputs,
    validate_video_split,
)

MODULE_B_SYSTEM_PROMPT = (
    "You are a temporal grounding model. Locate the requested instruction in the provided video clip. "
    "If the instruction attempt is visible, return JSON only as "
    "{\"relevant_windows\":[[\"start_seconds\",\"end_seconds\"]]} using seconds relative to the start of the clip. "
    "If the instruction attempt is not completed in the clip, return not completed."
)

os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen VL for Module B temporal grounding.")
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
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--bias", default="none")
    parser.add_argument("--use-rslora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loftq-config", default=None)
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
    parser.add_argument("--metric-for-best-model", default="eval_temporal/mean_iou")
    parser.add_argument("--greater-is-better", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--val-videos-per-task", type=int, default=2)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--regenerate-split", action="store_true")
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--completion-margin", type=float, default=0.25)
    parser.add_argument("--negative-to-positive-ratio", type=int, default=2)
    parser.add_argument("--keep-last-wait-windows", type=int, default=2)
    parser.add_argument("--module-b-pre-pad-ratio-min", type=float, default=0.25)
    parser.add_argument("--module-b-pre-pad-ratio-max", type=float, default=0.75)
    parser.add_argument("--module-b-post-pad-ratio-min", type=float, default=0.25)
    parser.add_argument("--module-b-post-pad-ratio-max", type=float, default=0.75)
    parser.add_argument("--include-incomplete-negatives", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--module-b-negative-ratio", type=float, default=0.25)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--vision-resize", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=6144)
    parser.add_argument("--eval-generation-max-samples", type=int, default=-1)
    parser.add_argument("--eval-generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", default="outputs/module_b_qwen35_lora_grounding")
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-examples", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--wandb-log-best-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-artifact-prefix", default="module-b")
    args = parser.parse_args()
    args.report_to = [item.strip() for item in args.report_to.split(",") if item.strip()]
    if args.max_videos <= 0:
        parser.error("--max-videos must be positive.")
    if args.keep_last_checkpoints < 0 or args.keep_best_checkpoints < 0:
        parser.error("checkpoint retention counts cannot be negative.")
    return args


def build_module_b_dataset(args: argparse.Namespace) -> EgoOopsModuleBDataset:
    provider = EgoOopsProvider(
        metadata_path=args.metadata,
        mistake_classes_path=args.mistake_classes,
        video_root=args.video_root,
        video_ids=set(args.video_ids) if args.video_ids else None,
        task_ids=set(args.task_ids) if args.task_ids else None,
        max_videos=args.max_videos,
        require_existing_videos=True,
    )
    module_a_config = ModuleAConfig(
        stride_seconds=args.stride_seconds,
        completion_margin=args.completion_margin,
        negative_to_positive_ratio=args.negative_to_positive_ratio,
        keep_last_wait_windows=args.keep_last_wait_windows,
        seed=args.seed,
    )
    module_b_config = ModuleBConfig(
        pre_pad_ratio_min=args.module_b_pre_pad_ratio_min,
        pre_pad_ratio_max=args.module_b_pre_pad_ratio_max,
        post_pad_ratio_min=args.module_b_post_pad_ratio_min,
        post_pad_ratio_max=args.module_b_post_pad_ratio_max,
        include_incomplete_negatives=args.include_incomplete_negatives,
        negative_ratio=args.module_b_negative_ratio,
        seed=args.seed,
    )
    return EgoOopsModuleBDataset(provider, module_a_config=module_a_config, config=module_b_config)


def default_split_file(args: argparse.Namespace) -> Path:
    return PROJECT_ROOT / "splits" / f"module_b_video_split_seed{args.seed}.json"


def load_or_create_video_split(dataset: EgoOopsModuleBDataset, args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    split_path = Path(args.split_file) if args.split_file else default_split_file(args)
    if split_path.exists() and not args.regenerate_split:
        with open(split_path, "r", encoding="utf-8") as file:
            split = json.load(file)
        validate_video_split(split, dataset)
        return split, split_path
    split = make_video_split(dataset, args)
    split["version"] = "module_b_video_split_v1"
    split["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as file:
        json.dump(split, file, indent=2, sort_keys=True)
        file.write("\n")
    return split, split_path


def to_unsloth_conversations(
    dataset: EgoOopsModuleBDataset,
    *,
    fps: float,
    min_frames: int,
    max_frames: int,
) -> list[dict[str, Any]]:
    return [
        training_example_to_conversation(example, fps=fps, min_frames=min_frames, max_frames=max_frames)
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
            {"role": "system", "content": [{"type": "text", "text": MODULE_B_SYSTEM_PROMPT}]},
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
        "metadata": example.metadata,
    }


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


def summarize_dataset(dataset: EgoOopsModuleBDataset) -> dict[str, Any]:
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
    dataset: EgoOopsModuleBDataset,
    conversations: list[dict[str, Any]],
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
        "num_conversations": len(conversations),
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


def parse_temporal_prediction(text: str) -> tuple[float, float] | None:
    cleaned = text.strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", cleaned)]
        if len(numbers) >= 2:
            return normalize_span(numbers[0], numbers[1])
        return None

    windows = payload.get("relevant_windows")
    if isinstance(windows, list) and windows:
        first = windows[0]
        if isinstance(first, list | tuple) and len(first) >= 2:
            return normalize_span(first[0], first[1])
    if "start_time" in payload and "end_time" in payload:
        return normalize_span(payload["start_time"], payload["end_time"])
    return None


def normalize_span(start: Any, end: Any) -> tuple[float, float] | None:
    try:
        parsed_start = float(start)
        parsed_end = float(end)
    except (TypeError, ValueError):
        return None
    if parsed_start < 0 or parsed_end < parsed_start:
        return None
    return parsed_start, parsed_end


def temporal_iou(pred: tuple[float, float], target: tuple[float, float]) -> float:
    start = max(pred[0], target[0])
    end = min(pred[1], target[1])
    intersection = max(0.0, end - start)
    union = max(pred[1], target[1]) - min(pred[0], target[0])
    return intersection / union if union > 0 else 0.0


def temporal_metrics(
    targets: list[tuple[float, float]],
    predictions: list[tuple[float, float] | None],
    labels: list[str] | None = None,
) -> dict[str, float]:
    labels = labels or ["LOCALIZE"] * len(targets)
    positives = [
        (target, pred)
        for target, pred, label in zip(targets, predictions, labels)
        if label != "NO_ACTION"
    ]
    negatives = [
        pred
        for pred, label in zip(predictions, labels)
        if label == "NO_ACTION"
    ]
    valid = [(target, pred) for target, pred in positives if pred is not None]
    invalid_count = len(positives) - len(valid)
    ious = [temporal_iou(pred, target) for target, pred in valid]
    metrics: dict[str, float] = {
        "eval_temporal/num_samples": float(len(targets)),
        "eval_temporal/num_positive_samples": float(len(positives)),
        "eval_temporal/num_no_action_samples": float(len(negatives)),
        "eval_temporal/invalid_rate": invalid_count / len(positives) if positives else 0.0,
        "eval_temporal/mean_iou": sum(ious) / len(ious) if ious else 0.0,
    }
    if negatives:
        no_action_correct = sum(1 for pred in negatives if pred is None)
        metrics["eval_temporal/no_action_accuracy"] = no_action_correct / len(negatives)
        metrics["eval_temporal/no_action_false_positive_rate"] = 1.0 - metrics["eval_temporal/no_action_accuracy"]
    else:
        metrics["eval_temporal/no_action_accuracy"] = 0.0
        metrics["eval_temporal/no_action_false_positive_rate"] = 0.0
    for threshold in (0.1, 0.3, 0.5):
        hits = sum(1 for value in ious if value >= threshold)
        metrics[f"eval_temporal/recall_at_{threshold:.1f}"] = hits / len(positives) if positives else 0.0
        metrics[f"eval_temporal/f1_at_{threshold:.1f}"] = (
            2 * hits / (len(valid) + len(positives)) if valid or positives else 0.0
        )
    if valid:
        start_errors = [abs(pred[0] - target[0]) for target, pred in valid]
        end_errors = [abs(pred[1] - target[1]) for target, pred in valid]
        center_errors = [abs(((pred[0] + pred[1]) / 2) - ((target[0] + target[1]) / 2)) for target, pred in valid]
        duration_errors = [abs((pred[1] - pred[0]) - (target[1] - target[0])) for target, pred in valid]
        metrics.update({
            "eval_temporal/start_mae": sum(start_errors) / len(start_errors),
            "eval_temporal/end_mae": sum(end_errors) / len(end_errors),
            "eval_temporal/center_mae": sum(center_errors) / len(center_errors),
            "eval_temporal/duration_mae": sum(duration_errors) / len(duration_errors),
        })
        for tolerance in (0.5, 1.0, 2.0):
            within = sum(
                1 for start_error, end_error in zip(start_errors, end_errors)
                if start_error <= tolerance and end_error <= tolerance
            )
            metrics[f"eval_temporal/boundary_within_{tolerance:.1f}s"] = within / len(positives)
    else:
        metrics.update({
            "eval_temporal/start_mae": 0.0,
            "eval_temporal/end_mae": 0.0,
            "eval_temporal/center_mae": 0.0,
            "eval_temporal/duration_mae": 0.0,
            "eval_temporal/boundary_within_0.5s": 0.0,
            "eval_temporal/boundary_within_1.0s": 0.0,
            "eval_temporal/boundary_within_2.0s": 0.0,
        })
    return metrics


class ModuleBTemporalMetricsCallback:
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
        predictions: list[tuple[float, float] | None] = []
        targets: list[tuple[float, float]] = []
        labels: list[str] = []
        import torch
        from unsloth import FastVisionModel

        FastVisionModel.for_inference(model)
        try:
            with torch.inference_mode():
                for example in self.eval_examples:
                    generated_text = self._generate_span(model, example)
                    parsed = parse_temporal_prediction(generated_text)
                    if parsed is not None:
                        parsed = (
                            example["window_start"] + parsed[0],
                            example["window_start"] + parsed[1],
                        )
                    predictions.append(parsed)
                    targets.append((float(example["gt_start"]), float(example["gt_end"])))
                    labels.append(str(example.get("label", "LOCALIZE")))
        finally:
            if was_training:
                FastVisionModel.for_training(model)
            release_cuda_memory()

        metrics = temporal_metrics(targets, predictions, labels)
        callback_metrics = kwargs.get("metrics")
        if isinstance(callback_metrics, dict):
            callback_metrics.update(metrics)
        self.trainer.log(metrics)
        return control

    def _generation_inputs(self, example: dict[str, Any]) -> dict[str, Any]:
        messages = [
            dict(message)
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

    def _generate_span(self, model: Any, example: dict[str, Any]) -> str:
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
    temporal_callback = ModuleBTemporalMetricsCallback(
        eval_examples=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        max_samples=args.eval_generation_max_samples,
        max_new_tokens=args.eval_generation_max_new_tokens,
    )
    callbacks.append(temporal_callback)
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
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_eval_dataset,
        data_collator=data_collator,
        args=SFTConfig(**trainer_args),
        callbacks=callbacks,
    )
    temporal_callback.trainer = trainer
    checkpoint_callback.trainer = trainer
    return trainer


def print_dry_run(
    dataset: EgoOopsModuleBDataset,
    train_conversations: list[dict[str, Any]],
    val_conversations: list[dict[str, Any]],
    split: dict[str, Any],
    split_path: Path,
    limit: int,
) -> None:
    print("Module B full dataset summary:")
    print(json.dumps(summarize_dataset(dataset), indent=2, ensure_ascii=False))
    print(f"Module B split file: {split_path}")
    print("Module B split video summary:")
    print(json.dumps({
        "train_video_ids": split["train_video_ids"],
        "val_video_ids": split["val_video_ids"],
        "tasks": split["tasks"],
    }, indent=2, ensure_ascii=False))
    print("Module B train summary:")
    print(json.dumps(summarize_conversations(train_conversations), indent=2, ensure_ascii=False))
    print("Module B validation summary:")
    print(json.dumps(summarize_conversations(val_conversations), indent=2, ensure_ascii=False))
    for index, conversation in enumerate(train_conversations[: max(0, limit)]):
        print(f"\nExample {index}:")
        print(json.dumps(conversation, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    dataset = build_module_b_dataset(args)
    conversations = to_unsloth_conversations(dataset, fps=args.fps, min_frames=args.min_frames, max_frames=args.max_frames)
    if not conversations:
        raise SystemExit("Module B dataset produced no conversations.")
    split, split_path = load_or_create_video_split(dataset, args)
    train_conversations, val_conversations = split_conversations(conversations, split)
    if args.dry_run:
        print_dry_run(dataset, train_conversations, val_conversations, split, split_path, args.print_examples)
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
    print("Module B full dataset summary:")
    print(json.dumps(summarize_dataset(dataset), indent=2, ensure_ascii=False))
    print(f"Module B split file: {split_path}")
    print("Module B train summary:")
    print(json.dumps(summarize_conversations(train_conversations), indent=2, ensure_ascii=False))
    print("Module B validation summary:")
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
