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
from ..time_tokens import render_time_span_target
from ..video import video_duration_seconds


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
            instruction = "undefined / extra mistake action"
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


def _module_b_prompt(instruction: str) -> str:
    return (
        f"Instruction: {instruction}\n"
        "Identify the precise start and end boundaries."
    )


def _module_c_prompt(instruction: str) -> str:
    return f"Instruction: {instruction}\nAnalyze the execution."


def _procedural_segments(record: VideoRecord) -> list[Segment]:
    return [
        segment for segment in record.segments
        if segment.instruction_index >= 0 and segment.duration >= 0.5
    ]


def build_module_a_windows(
    records: Iterable[VideoRecord],
    *,
    config: ModuleAConfig,
) -> list[WindowSpec]:
    rng = random.Random(config.seed)
    all_windows: list[WindowSpec] = []

    for record in records:
        video_duration = video_duration_seconds(record.video_path)
        if video_duration <= 0:
            video_duration = max((segment.end for segment in record.segments), default=0.0)

        completed: list[str] = []
        pending_waits: list[WindowSpec] = []
        positives = 0
        last_reset_time = 0.0

        procedural = _procedural_segments(record)
        for step_order, segment in enumerate(procedural):
            buffer_start = max(0.0, last_reset_time)
            current_time = min(video_duration, buffer_start + config.stride_seconds)
            completion_deadline = min(
                video_duration,
                segment.end + segment.duration * config.completion_margin,
            )

            pending = tuple(
                item.instruction
                for item in procedural[step_order + 1 :]
            )

            while current_time < completion_deadline:
                label = "[WAIT]" if current_time < segment.end else "[COMPLETE]"
                window = WindowSpec(
                    video_path=record.video_path,
                    video_id=record.video_id,
                    task_id=record.task_id,
                    step_index=segment.instruction_index,
                    current_step=segment.instruction,
                    window_start=buffer_start,
                    window_end=current_time,
                    gt_start=segment.start,
                    gt_end=segment.end,
                    label=label,
                    completed_steps=tuple(completed),
                    pending_steps=pending,
                )
                if label == "[WAIT]":
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
                positives += 1
                pending_waits = []
                completed.append(segment.instruction)
                last_reset_time = segment.end
                break

            else:
                complete_time = min(video_duration, max(segment.end, current_time))
                window = WindowSpec(
                    video_path=record.video_path,
                    video_id=record.video_id,
                    task_id=record.task_id,
                    step_index=segment.instruction_index,
                    current_step=segment.instruction,
                    window_start=buffer_start,
                    window_end=complete_time,
                    gt_start=segment.start,
                    gt_end=segment.end,
                    label="[COMPLETE]",
                    completed_steps=tuple(completed),
                    pending_steps=pending,
                )
                all_windows.extend(_downsample_waits(
                    pending_waits,
                    rng=rng,
                    max_waits=max(0, config.negative_to_positive_ratio),
                    keep_last=config.keep_last_wait_windows,
                ))
                all_windows.append(window)
                positives += 1
                pending_waits = []
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
        self.records = list(provider.iter_video_records())
        self.windows = build_module_a_windows(self.records, config=self.config)
        self.examples = [self._window_to_example(window) for window in self.windows]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self.examples[index]

    @staticmethod
    def _window_to_example(window: WindowSpec) -> TrainingExample:
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
        self.module_a_config = module_a_config or ModuleAConfig()
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
            if window.label == "[COMPLETE]" and window.gt_start is not None and window.gt_end is not None
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
            target = render_time_span_target(
                gt_start=window.gt_start,
                gt_end=window.gt_end,
                window_start=start,
                window_end=end,
                config=self.time_config,
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
                metadata={"instruction": instruction},
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
            if window.label == "[WAIT]"
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
                target_text=TimeTokenConfig().no_action_token,
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
                if segment.duration < 0.5:
                    continue
                jitter = rng.uniform(self.config.jitter_ratio_min, self.config.jitter_ratio_max) * segment.duration
                start = max(0.0, segment.start + rng.uniform(-jitter, jitter))
                end = min(video_duration, segment.end + rng.uniform(-jitter, jitter))
                if end - start < 0.5:
                    start, end = segment.start, segment.end
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
                    metadata={"instruction": segment.instruction, "error_label": segment.error_label},
                ))
        return examples


def _instruction_lookup(records: list[VideoRecord]) -> dict[tuple[str, int], str]:
    lookup = {}
    for record in records:
        for segment in _procedural_segments(record):
            lookup[(record.video_id, segment.instruction_index)] = segment.instruction
    return lookup


def _reasoning_for_segment(segment: Segment) -> str:
    if segment.caption:
        return segment.caption
    if segment.is_mistake:
        if segment.error_label:
            return f"The segment is labeled as {segment.error_label}."
        return "The segment is labeled as an undefined or extra mistake action."
    return "The action follows the expected instruction."
