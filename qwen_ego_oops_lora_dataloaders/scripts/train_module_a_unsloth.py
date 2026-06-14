from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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

MODULE_A_SYSTEM_PROMPT = "You are an online procedural step completion detector."
DEFAULT_OUTPUT_DIR = "/nvcr/users/afeldman/omd/module_a_qwen35_2b_lora"


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

    # Module A simulation
    parser.add_argument("--stride-seconds", type=float, default=5.0)
    parser.add_argument("--completion-margin", type=float, default=0.10)
    parser.add_argument("--negative-to-positive-ratio", type=int, default=2)
    parser.add_argument("--keep-last-wait-windows", type=int, default=2)

    # Model / LoRA
    parser.add_argument("--model-name", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-16bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--finetune-vision-layers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--finetune-language-layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune-attention-modules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune-mlp-modules", action=argparse.BooleanOptionalAction, default=True)

    # Training
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--dataset-num-proc", type=int, default=1)

    # Validation / execution mode
    parser.add_argument("--dry-run", action="store_true", help="Build dataset and print examples only.")
    parser.add_argument("--print-examples", type=int, default=1)
    return parser.parse_args()


def is_bf16_available() -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


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


def to_unsloth_conversations(dataset: EgoOopsModuleADataset, fps: float) -> list[dict[str, Any]]:
    return [training_example_to_conversation(example, fps=fps) for example in dataset]


def training_example_to_conversation(example: TrainingExample, *, fps: float) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": MODULE_A_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(example.video_path),
                        "fps": fps,
                    },
                    {
                        "type": "text",
                        "text": example.prompt_text,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": example.target_text,
            },
        ],
        "video_id": example.video_id,
        "task_id": example.task_id,
        "step_index": example.step_index,
        "window_start": example.window_start,
        "window_end": example.window_end,
        "target": example.target_text,
    }


def summarize_module_a(dataset: EgoOopsModuleADataset) -> dict[str, Any]:
    counts = Counter(example.target_text for example in dataset)
    videos = {example.video_id for example in dataset}
    return {
        "num_examples": len(dataset),
        "num_videos": len(videos),
        "label_counts": dict(sorted(counts.items())),
    }


def print_dry_run(dataset: EgoOopsModuleADataset, conversations: list[dict[str, Any]], limit: int) -> None:
    print("Module A dataset summary:")
    print(json.dumps(summarize_module_a(dataset), indent=2, ensure_ascii=False))
    for index, conversation in enumerate(conversations[: max(0, limit)]):
        print(f"\nExample {index}:")
        print(json.dumps(conversation, indent=2, ensure_ascii=False))


def load_unsloth_model(args: argparse.Namespace) -> tuple[Any, Any]:
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=args.load_in_16bit,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
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
        modules_to_save=["lm_head", "embed_tokens"],
    )
    return model, tokenizer


def build_trainer(
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Any:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    FastVisionModel.for_training(model)
    bf16 = is_bf16_available()
    hf_train_dataset = Dataset.from_list(train_dataset)
    trainer_args = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        report_to=args.report_to,
        dataset_num_proc=args.dataset_num_proc,
        max_seq_length=args.max_seq_length,
        fp16=not bf16,
        bf16=bf16,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    if args.num_train_epochs is not None:
        trainer_args["num_train_epochs"] = args.num_train_epochs
    else:
        trainer_args["max_steps"] = args.max_steps

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_train_dataset,
        data_collator=UnslothVisionDataCollator(
            model,
            tokenizer,
            train_on_responses_only=True,
            completion_only_loss=True,
        ),
        args=SFTConfig(**trainer_args),
    )


def save_outputs(model: Any, tokenizer: Any, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Saved LoRA adapter and tokenizer to: {output_path}")


def main() -> None:
    args = parse_args()
    dataset = build_module_a_dataset(args)
    conversations = to_unsloth_conversations(dataset, fps=args.fps)

    if not conversations:
        raise SystemExit("Module A dataset produced no conversations.")
    if args.dry_run:
        print_dry_run(dataset, conversations, args.print_examples)
        return

    print("Module A dataset summary:")
    print(json.dumps(summarize_module_a(dataset), indent=2, ensure_ascii=False))

    model, tokenizer = load_unsloth_model(args)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=conversations,
        args=args,
    )
    trainer.train()
    save_outputs(model, tokenizer, args.output_dir)


if __name__ == "__main__":
    main()
