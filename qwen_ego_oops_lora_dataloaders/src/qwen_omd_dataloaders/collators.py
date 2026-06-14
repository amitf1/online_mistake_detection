from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .config import VideoSamplingConfig
from .schema import TrainingExample
from .video import sample_video_window


SYSTEM_PROMPTS = {
    "A": "You are an online procedural step completion detector.",
    "B": "You are a temporal action localization assistant.",
    "C": "You are a procedural mistake detection and reasoning assistant.",
}


@dataclass
class BatchDebugItem:
    video_id: str
    task_id: str
    module: str
    step_index: int
    window_start: float
    window_end: float
    frame_timestamps: list[float]
    prompt_text: str
    target_text: str


class QwenChatMLCollator:
    """Collate TrainingExample objects into supervised Qwen video-text batches."""

    module: str = ""

    def __init__(
        self,
        *,
        processor: Any,
        video_config: VideoSamplingConfig | None = None,
        include_metadata: bool = False,
    ) -> None:
        self.processor = processor
        self.video_config = video_config or VideoSamplingConfig()
        self.include_metadata = include_metadata

    def __call__(self, examples: list[TrainingExample]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")

        texts: list[str] = []
        videos: list[Any] = []
        debug_items: list[BatchDebugItem] = []
        prompt_lengths: list[int] = []

        for example in examples:
            frames, timestamps = sample_video_window(
                video_path=example.video_path,
                start_time=example.window_start,
                end_time=example.window_end,
                config=self.video_config,
            )
            videos.append(frames)
            prompt_text = self._render_prompt(example)
            full_text = self._render_full(example)
            texts.append(full_text)
            prompt_lengths.append(self._prompt_token_length(prompt_text, frames))
            debug_items.append(BatchDebugItem(
                video_id=example.video_id,
                task_id=example.task_id,
                module=example.module,
                step_index=example.step_index,
                window_start=example.window_start,
                window_end=example.window_end,
                frame_timestamps=timestamps,
                prompt_text=example.prompt_text,
                target_text=example.target_text,
            ))

        batch = self.processor(
            text=texts,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        pad_token_id = self._pad_token_id()
        for row, prompt_len in enumerate(prompt_lengths):
            labels[row, : min(prompt_len, labels.shape[1])] = -100
        if pad_token_id is not None:
            labels[batch["input_ids"] == pad_token_id] = -100
        batch["labels"] = labels
        if self.include_metadata:
            batch["debug"] = debug_items
        return batch

    def _render_prompt(self, example: TrainingExample) -> str:
        messages = self._messages(example, include_answer=False)
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _render_full(self, example: TrainingExample) -> str:
        messages = self._messages(example, include_answer=True)
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def _messages(self, example: TrainingExample, *, include_answer: bool) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPTS.get(example.module, "You are a video assistant."),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(example.video_path),
                        "fps": self.video_config.sample_fps,
                    },
                    {"type": "text", "text": example.prompt_text},
                ],
            },
        ]
        if include_answer:
            messages.append({"role": "assistant", "content": example.target_text})
        return messages

    def _prompt_token_length(self, prompt_text: str, frames: Any) -> int:
        prompt_inputs = self.processor(
            text=[prompt_text],
            videos=[frames],
            padding=False,
            return_tensors="pt",
        )
        return int(prompt_inputs["input_ids"].shape[1])

    def _pad_token_id(self) -> int | None:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            return getattr(tokenizer, "pad_token_id", None)
        return getattr(self.processor, "pad_token_id", None)


class ModuleACollator(QwenChatMLCollator):
    module = "A"


class ModuleBCollator(QwenChatMLCollator):
    module = "B"


class ModuleCCollator(QwenChatMLCollator):
    module = "C"
