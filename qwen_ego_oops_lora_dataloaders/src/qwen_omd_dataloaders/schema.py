from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ModuleName = Literal["A", "B", "C"]


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    instruction_index: int
    instruction: str
    is_mistake: bool
    error_label: str | None = None
    caption: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class VideoRecord:
    dataset: str
    task_id: str
    video_id: str
    video_path: Path
    instructions: list[str]
    segments: list[Segment]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WindowSpec:
    video_path: Path
    video_id: str
    task_id: str
    step_index: int
    current_step: str
    window_start: float
    window_end: float
    gt_start: float | None
    gt_end: float | None
    label: str
    completed_steps: tuple[str, ...] = ()
    pending_steps: tuple[str, ...] = ()

    @property
    def duration(self) -> float:
        return max(0.0, self.window_end - self.window_start)


@dataclass(frozen=True)
class TrainingExample:
    module: ModuleName
    source_dataset: str
    video_path: Path
    video_id: str
    task_id: str
    step_index: int
    window_start: float
    window_end: float
    prompt_text: str
    target_text: str
    gt_start: float | None = None
    gt_end: float | None = None
    label: str | None = None
    mistake: bool | None = None
    reasoning: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
