from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    class Dataset:  # type: ignore[no-redef]
        """Import-time fallback so metadata smoke tests work before torch install."""
        pass

from ..config import ModuleAConfig, ModuleBConfig, ModuleCConfig, TimeTokenConfig
from ..schema import Segment, TrainingExample, VideoRecord, WindowSpec
from ..time_tokens import render_seconds_span_target
from ..video import video_duration_seconds

EXTRA_STEP_INSTRUCTION = "undefined / extra step"

TASK_DESCRIPTIONS: dict[str, str] = {
    "blacklight": "ionic reaction / fluorescence experiment with highlighters, detergent, and a black light",
    "cardboard": "cardboard box crafting procedure (cutting, folding, gluing, assembling)",
    "electronics": "electronics circuit assembly with battery box, switches, motor, and lamp",
    "ion": "metal ion reactivity experiment on a microplate with copper, zinc, and magnesium",
    "tsumiki": "block stacking (tsumiki) assembly with colored cubes and prisms",
}

MODULE_A_STEP_ID_NONE_LETTER = "A"
MODULE_A_STEP_ID_EXTRA_LETTER = "B"
MODULE_A_STEP_ID_FIRST_INSTRUCTION_LETTER = "C"


class EgoOopsProvider:
    """Normalize EgoOops metadata into records shared by all module datasets."""

    def __init__(
        self,
        *,
        metadata_path: str | Path,
        mistake_classes_path: str | Path,
        video_root: str | Path,
        video_ids: set[str] | None = None,
        task_ids: set[str] | None = None,
        max_videos: int | None = None,
        require_existing_videos: bool = True,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.mistake_classes_path = Path(mistake_classes_path)
        self.video_root = Path(video_root)
        self.video_ids = video_ids
        self.task_ids = task_ids
        self.max_videos = max_videos
        self.require_existing_videos = require_existing_videos

        with open(self.metadata_path, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)
        with open(self.mistake_classes_path, "r", encoding="utf-8") as file:
            self.mistake_classes = list(json.load(file))

    def iter_video_records(self) -> Iterable[VideoRecord]:
        yielded = 0
        instructions_by_task = self.metadata.get("instructions", {})
        for video_meta in self.metadata.get("videos", []):
            video_id = str(video_meta["video_id"])
            task_id = str(video_meta["task_id"])
            if self.video_ids and video_id not in self.video_ids:
                continue
            if self.task_ids and task_id not in self.task_ids:
                continue

            video_path = self.video_root / task_id / f"{video_id}.MP4"
            if self.require_existing_videos and not video_path.exists():
                continue

            instructions = [str(item) for item in instructions_by_task.get(task_id, [])]
            segments = [
                self._normalize_segment(raw_segment, instructions)
                for raw_segment in video_meta.get("segments", [])
            ]
            segments.sort(key=lambda item: (item.start, item.end))

            yield VideoRecord(
                dataset="ego_oops",
                task_id=task_id,
                video_id=video_id,
                video_path=video_path,
                instructions=instructions,
                segments=segments,
                raw=video_meta,
            )
            yielded += 1
            if self.max_videos is not None and yielded >= self.max_videos:
                break

    def _normalize_segment(self, raw_segment: dict[str, Any], instructions: list[str]) -> Segment:
        instruction_index = int(raw_segment.get("instruction", -1))
        labels = list(raw_segment.get("labels", []))
        is_mistake = instruction_index == -1 or bool(labels)
        label_names = [self._label_name(label) for label in labels]
        if instruction_index == -1:
            instruction = EXTRA_STEP_INSTRUCTION
        elif 0 <= instruction_index < len(instructions):
            instruction = instructions[instruction_index]
        else:
            instruction = f"unknown instruction {instruction_index}"

        return Segment(
            start=float(raw_segment["startTime"]),
            end=float(raw_segment["endTime"]),
            instruction_index=instruction_index,
            instruction=instruction,
            is_mistake=is_mistake,
            error_label=", ".join(label_names) if label_names else None,
            caption=str(raw_segment.get("caption", "")).strip(),
            raw=raw_segment,
        )

    def _label_name(self, label: Any) -> str:
        if isinstance(label, int) and 0 <= label < len(self.mistake_classes):
            return self.mistake_classes[label]
        return str(label)


def task_description(task_id: str) -> str:
    return TASK_DESCRIPTIONS.get(task_id, f"procedural task '{task_id}'")


def module_a_step_id_letter_for_instruction_index(instruction_index: int) -> str:
    if instruction_index < 0:
        return MODULE_A_STEP_ID_EXTRA_LETTER
    return chr(ord(MODULE_A_STEP_ID_FIRST_INSTRUCTION_LETTER) + instruction_index)


def module_a_instruction_for_step_id_letter(
    letter: str,
    instructions: list[str] | tuple[str, ...],
) -> str | None:
    normalized = letter.strip().upper()[:1]
    if not normalized or not ("A" <= normalized <= "Z"):
        return None
    if normalized == MODULE_A_STEP_ID_NONE_LETTER:
        return None
    if normalized == MODULE_A_STEP_ID_EXTRA_LETTER:
        return EXTRA_STEP_INSTRUCTION
    index = ord(normalized) - ord(MODULE_A_STEP_ID_FIRST_INSTRUCTION_LETTER)
    if 0 <= index < len(instructions):
        return instructions[index]
    return None


def is_module_a_completion_label(label: str, *, label_mode: str) -> bool:
    if label_mode == "legacy":
        return label.strip().upper() == "COMPLETE"
    return label.strip().upper()[:1] not in ("", MODULE_A_STEP_ID_NONE_LETTER)


def _module_a_prompt(
    *,
    completed_steps: Iterable[str],
    current_step: str,
    pending_steps: Iterable[str],
) -> str:
    return (
        "Completed Steps: "
        f"{json.dumps(list(completed_steps), ensure_ascii=False)}\n"
        f"Current Step: {current_step}\n"
        "Pending Steps: "
        f"{json.dumps(list(pending_steps), ensure_ascii=False)}\n"
        "Status?"
    )


def _module_a_step_id_prompt(
    *,
    task_id: str,
    instructions: Iterable[str],
    completed_steps: Iterable[str],
    mistake_classes: Iterable[str],
) -> str:
    instruction_list = list(instructions)
    completed = list(completed_steps)
    lines = [
        (
            f'In the following video, the person is doing a task called: "{task_id}". '
            f"Description of the task: {task_description(task_id)}."
        ),
    ]
    if completed:
        lines.append("The following steps were already done:")
        lines.extend(f"- {step}" for step in completed)
    else:
        lines.append("No steps were done yet.")
    lines.append(
        "You need to identify which of the instruction steps is being attempted. "
        "Take into consideration that mistakes can be done such as:"
    )
    lines.extend(f"- {name}" for name in mistake_classes)
    lines.append(
        "Also consider extra steps not in the formal directions. "
        "Which of the following steps is the person best likely just finished attempting "
        "from what is visible in the clip?"
    )
    lines.append(f"{MODULE_A_STEP_ID_NONE_LETTER}. NONE - no step was completed / truncated step")
    lines.append(
        f"{MODULE_A_STEP_ID_EXTRA_LETTER}. EXTRA - a completed extra step not in the list below"
    )
    for index, instruction in enumerate(instruction_list):
        letter = module_a_step_id_letter_for_instruction_index(index)
        lines.append(f"{letter}. {instruction}")
    return "\n".join(lines)


def _module_b_prompt(instruction: str) -> str:
    return (
        f"Instruction: {instruction}\n"
        "Identify the precise start and end boundaries for the attempt at this instruction in the provided video clip. "
        "The attempt may include mistakes or deviate from the exact instruction wording.\n"
        "If the attempt is visible, return JSON only as "
        "{\"relevant_windows\":[[\"start_seconds\",\"end_seconds\"]]} using seconds relative to the start of this clip. "
        "If the attempt is not completed in the clip, return exactly: not completed"
    )


def _module_c_prompt(instruction: str) -> str:
    return (
        f"Instruction: {instruction}\n"
        "Watch the bounded video clip and decide whether the visible execution contains an EgoOops-style "
        "mistake relative to the written instruction.\n"
        "A mistake includes using the wrong object, tool, color, material, or part; using an object in the "
        "wrong way; accidental or unintended actions; self-correction after an error; wrong timing, portion, "
        "or order inside the step; or any other visible deviation from the instruction.\n"
        "Do not mark ordinary execution variation as a mistake: brief pauses, repeated grasping, small hand "
        "adjustments, or repositioning are acceptable when the person is still following the instruction.\n"
        "Return JSON only with keys mistake and reasoning."
    )


def _procedural_segments(record: VideoRecord) -> list[Segment]:
    return [
        segment for segment in record.segments
        if segment.instruction_index >= 0 and segment.duration >= 0.5
    ]


def _module_a_attempt_segments(record: VideoRecord, *, label_mode: str) -> list[Segment]:
    if label_mode == "legacy":
        return _procedural_segments(record)
    return [segment for segment in record.segments if segment.duration >= 0.5]


def build_module_a_windows(
    records: Iterable[VideoRecord],
    *,
    config: ModuleAConfig,
) -> list[WindowSpec]:
    rng = random.Random(config.seed)
    all_windows: list[WindowSpec] = []
    label_mode = config.label_mode

    for record in records:
        video_duration = video_duration_seconds(record.video_path)
        if video_duration <= 0:
            video_duration = max((segment.end for segment in record.segments), default=0.0)

        completed: list[str] = []
        pending_waits: list[WindowSpec] = []
        last_reset_time = 0.0
        all_instructions = tuple(record.instructions)
        attempts = _module_a_attempt_segments(record, label_mode=label_mode)

        for step_order, segment in enumerate(attempts):
            buffer_start = max(0.0, last_reset_time)
            current_time = min(video_duration, buffer_start + config.stride_seconds)
            completion_deadline = min(
                video_duration,
                segment.end + segment.duration * config.completion_margin,
            )

            if label_mode == "legacy":
                pending = tuple(item.instruction for item in attempts[step_order + 1 :])
                incomplete_label = "WAIT"
                complete_label = "COMPLETE"
                current_step = segment.instruction
            else:
                pending = tuple(
                    item.instruction
                    for item in attempts[step_order + 1 :]
                    if item.instruction_index >= 0
                )
                incomplete_label = MODULE_A_STEP_ID_NONE_LETTER
                complete_label = module_a_step_id_letter_for_instruction_index(segment.instruction_index)
                current_step = (
                    EXTRA_STEP_INSTRUCTION
                    if segment.instruction_index < 0
                    else segment.instruction
                )

            emitted_complete = False
            while current_time < completion_deadline:
                label = incomplete_label if current_time < segment.end else complete_label
                window = WindowSpec(
                    video_path=record.video_path,
                    video_id=record.video_id,
                    task_id=record.task_id,
                    step_index=segment.instruction_index,
                    current_step=current_step,
                    window_start=buffer_start,
                    window_end=current_time,
                    gt_start=segment.start,
                    gt_end=segment.end,
                    label=label,
                    completed_steps=tuple(completed),
                    pending_steps=pending,
                    all_instructions=all_instructions,
                    label_mode=label_mode,
                )
                if label == incomplete_label:
                    pending_waits.append(window)
                    current_time += config.stride_seconds
                    continue

                all_windows.extend(_downsample_waits(
                    pending_waits,
                    rng=rng,
                    max_waits=max(0, config.negative_to_positive_ratio),
                    keep_last=config.keep_last_wait_windows,
                ))
                all_windows.append(window)
                pending_waits = []
                if segment.instruction_index >= 0:
                    completed.append(segment.instruction)
                last_reset_time = segment.end
                emitted_complete = True
                break

            if not emitted_complete:
                complete_time = min(video_duration, max(segment.end, current_time))
                window = WindowSpec(
                    video_path=record.video_path,
                    video_id=record.video_id,
                    task_id=record.task_id,
                    step_index=segment.instruction_index,
                    current_step=current_step,
                    window_start=buffer_start,
                    window_end=complete_time,
                    gt_start=segment.start,
                    gt_end=segment.end,
                    label=complete_label,
                    completed_steps=tuple(completed),
                    pending_steps=pending,
                    all_instructions=all_instructions,
                    label_mode=label_mode,
                )
                all_windows.extend(_downsample_waits(
                    pending_waits,
                    rng=rng,
                    max_waits=max(0, config.negative_to_positive_ratio),
                    keep_last=config.keep_last_wait_windows,
                ))
                all_windows.append(window)
                pending_waits = []
                if segment.instruction_index >= 0:
                    completed.append(segment.instruction)
                last_reset_time = segment.end

    return all_windows


def _downsample_waits(
    waits: list[WindowSpec],
    *,
    rng: random.Random,
    max_waits: int,
    keep_last: int,
) -> list[WindowSpec]:
    if not waits or max_waits <= 0:
        return []
    keep: list[WindowSpec] = waits[-keep_last:] if keep_last > 0 else []
    remaining = waits[: max(0, len(waits) - len(keep))]
    slots = max(0, max_waits - len(keep))
    if slots and remaining:
        if len(remaining) <= slots:
            keep = remaining + keep
        else:
            keep = rng.sample(remaining, slots) + keep
    return sorted(keep, key=lambda item: item.window_end)


class EgoOopsModuleADataset(Dataset):
    def __init__(self, provider: EgoOopsProvider, config: ModuleAConfig | None = None) -> None:
        self.config = config or ModuleAConfig()
        self.provider = provider
        self.mistake_classes = list(provider.mistake_classes)
        self.records = list(provider.iter_video_records())
        self.windows = build_module_a_windows(self.records, config=self.config)
        self.examples = [self._window_to_example(window) for window in self.windows]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self.examples[index]

    def _window_to_example(self, window: WindowSpec) -> TrainingExample:
        if window.label_mode == "step_id":
            prompt = _module_a_step_id_prompt(
                task_id=window.task_id,
                instructions=window.all_instructions or (),
                completed_steps=window.completed_steps,
                mistake_classes=self.mistake_classes,
            )
        else:
            prompt = _module_a_prompt(
                completed_steps=window.completed_steps,
                current_step=window.current_step,
                pending_steps=window.pending_steps,
            )
        return TrainingExample(
            module="A",
            source_dataset="ego_oops",
            video_path=window.video_path,
            video_id=window.video_id,
            task_id=window.task_id,
            step_index=window.step_index,
            window_start=window.window_start,
            window_end=window.window_end,
            gt_start=window.gt_start,
            gt_end=window.gt_end,
            prompt_text=prompt,
            target_text=window.label,
            label=window.label,
            metadata={
                "completed_steps": list(window.completed_steps),
                "current_step": window.current_step,
                "pending_steps": list(window.pending_steps),
                "all_instructions": list(window.all_instructions),
                "label_mode": window.label_mode,
                "mistake_classes": list(self.mistake_classes),
            },
        )


class EgoOopsModuleBDataset(Dataset):
    def __init__(
        self,
        provider: EgoOopsProvider,
        module_a_config: ModuleAConfig | None = None,
        config: ModuleBConfig | None = None,
        time_config: TimeTokenConfig | None = None,
    ) -> None:
        base = module_a_config or ModuleAConfig()
        # Module B localizes GT procedural attempts; keep legacy COMPLETE windows.
        self.module_a_config = ModuleAConfig(
            stride_seconds=base.stride_seconds,
            completion_margin=base.completion_margin,
            negative_to_positive_ratio=base.negative_to_positive_ratio,
            keep_last_wait_windows=base.keep_last_wait_windows,
            seed=base.seed,
            label_mode="legacy",
        )
        self.config = config or ModuleBConfig()
        self.time_config = time_config or TimeTokenConfig()
        self.records = list(provider.iter_video_records())
        self.examples = self._build_examples()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self.examples[index]

    def _build_examples(self) -> list[TrainingExample]:
        rng = random.Random(self.config.seed)
        examples: list[TrainingExample] = []
        complete_windows = [
            window for window in build_module_a_windows(self.records, config=self.module_a_config)
            if window.label == "COMPLETE" and window.gt_start is not None and window.gt_end is not None
        ]
        instruction_lookup = _instruction_lookup(self.records)

        for window in complete_windows:
            duration = max(0.5, window.gt_end - window.gt_start)
            pre = rng.uniform(self.config.pre_pad_ratio_min, self.config.pre_pad_ratio_max) * duration
            post = rng.uniform(self.config.post_pad_ratio_min, self.config.post_pad_ratio_max) * duration
            video_duration = video_duration_seconds(window.video_path)
            if video_duration <= 0:
                video_duration = max(window.window_end, window.gt_end)
            start = max(0.0, window.gt_start - pre)
            end = min(video_duration, max(window.window_end, window.gt_end + post))
            if end - start < 0.5:
                continue
            instruction = instruction_lookup.get((window.video_id, window.step_index), f"step {window.step_index}")
            target = render_seconds_span_target(
                gt_start=window.gt_start,
                gt_end=window.gt_end,
                window_start=start,
            )
            examples.append(TrainingExample(
                module="B",
                source_dataset="ego_oops",
                video_path=window.video_path,
                video_id=window.video_id,
                task_id=window.task_id,
                step_index=window.step_index,
                window_start=start,
                window_end=end,
                gt_start=window.gt_start,
                gt_end=window.gt_end,
                prompt_text=_module_b_prompt(instruction),
                target_text=target,
                label="LOCALIZE",
                metadata={
                    "instruction": instruction,
                    "clip_relative_gt_start": window.gt_start - start,
                    "clip_relative_gt_end": window.gt_end - start,
                    "original_gt_start": window.gt_start,
                    "original_gt_end": window.gt_end,
                },
            ))

        if self.config.include_incomplete_negatives:
            examples.extend(self._build_incomplete_negatives(complete_windows, instruction_lookup, rng))
        return examples

    def _build_incomplete_negatives(
        self,
        complete_windows: list[WindowSpec],
        instruction_lookup: dict[tuple[str, int], str],
        rng: random.Random,
    ) -> list[TrainingExample]:
        wait_windows = [
            window for window in build_module_a_windows(self.records, config=self.module_a_config)
            if window.label == "WAIT"
        ]
        k = min(len(wait_windows), round(len(complete_windows) * self.config.negative_ratio))
        selected = rng.sample(wait_windows, k) if k > 0 else []
        examples = []
        for window in selected:
            instruction = instruction_lookup.get((window.video_id, window.step_index), f"step {window.step_index}")
            examples.append(TrainingExample(
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
                label="NO_ACTION",
                metadata={"instruction": instruction},
            ))
        return examples


class EgoOopsModuleCDataset(Dataset):
    def __init__(
        self,
        provider: EgoOopsProvider,
        config: ModuleCConfig | None = None,
    ) -> None:
        self.config = config or ModuleCConfig()
        self.records = list(provider.iter_video_records())
        self.examples = self._build_examples()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self.examples[index]

    def _build_examples(self) -> list[TrainingExample]:
        rng = random.Random(self.config.seed)
        examples: list[TrainingExample] = []
        for record in self.records:
            video_duration = video_duration_seconds(record.video_path)
            if video_duration <= 0:
                video_duration = max((segment.end for segment in record.segments), default=0.0)
            for segment in record.segments:
                if segment.duration < self.config.min_duration_seconds:
                    continue
                jitter = rng.uniform(self.config.jitter_ratio_min, self.config.jitter_ratio_max) * segment.duration
                start = max(0.0, segment.start + rng.uniform(-jitter, jitter))
                end = min(video_duration, segment.end + rng.uniform(-jitter, jitter))
                if end - start < self.config.min_duration_seconds:
                    start, end = segment.start, segment.end
                if end - start < self.config.min_duration_seconds:
                    continue
                reasoning = _reasoning_for_segment(segment)
                target = json.dumps(
                    {"mistake": segment.is_mistake, "reasoning": reasoning},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                examples.append(TrainingExample(
                    module="C",
                    source_dataset="ego_oops",
                    video_path=record.video_path,
                    video_id=record.video_id,
                    task_id=record.task_id,
                    step_index=segment.instruction_index,
                    window_start=start,
                    window_end=end,
                    gt_start=segment.start,
                    gt_end=segment.end,
                    prompt_text=_module_c_prompt(segment.instruction),
                    target_text=target,
                    label="MISTAKE" if segment.is_mistake else "CORRECT",
                    mistake=segment.is_mistake,
                    reasoning=reasoning,
                    metadata={
                        "instruction": segment.instruction,
                        "error_label": segment.error_label,
                        "mistake_labels": _split_error_labels(segment.error_label),
                        "raw_labels": list(segment.raw.get("labels", [])),
                        "caption": segment.caption,
                        "is_undefined_instruction": segment.instruction_index == -1,
                        "clip_relative_gt_start": segment.start - start,
                        "clip_relative_gt_end": segment.end - start,
                        "original_gt_start": segment.start,
                        "original_gt_end": segment.end,
                    },
                ))
        return examples


def _instruction_lookup(records: list[VideoRecord]) -> dict[tuple[str, int], str]:
    lookup = {}
    for record in records:
        for segment in _procedural_segments(record):
            lookup[(record.video_id, segment.instruction_index)] = segment.instruction
    return lookup


def _reasoning_for_segment(segment: Segment) -> str:
    if segment.is_mistake:
        if segment.caption:
            return segment.caption
        if segment.error_label:
            return f"The execution deviates from the instruction and is labeled as {segment.error_label}."
        return "The segment is labeled as an undefined or extra step."
    return "The observed action follows the instructed step without a visible procedural deviation."


def _split_error_labels(error_label: str | None) -> list[str]:
    if not error_label:
        return []
    return [item.strip() for item in error_label.split(",") if item.strip()]
