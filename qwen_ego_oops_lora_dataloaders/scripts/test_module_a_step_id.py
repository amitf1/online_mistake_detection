from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from qwen_omd_dataloaders.config import ModuleAConfig  # noqa: E402
from qwen_omd_dataloaders.datasets.ego_oops import (  # noqa: E402
    EXTRA_STEP_INSTRUCTION,
    MODULE_A_STEP_ID_EXTRA_LETTER,
    MODULE_A_STEP_ID_NONE_LETTER,
    build_module_a_windows,
    is_module_a_completion_label,
    module_a_instruction_for_step_id_letter,
    module_a_step_id_letter_for_instruction_index,
)
from qwen_omd_dataloaders.schema import Segment, VideoRecord  # noqa: E402
from train_module_a_unsloth import extract_module_a_label  # noqa: E402


def fake_record(
    *,
    instructions: list[str],
    segments: list[Segment],
    video_id: str = "vid",
    task_id: str = "electronics",
) -> VideoRecord:
    return VideoRecord(
        dataset="ego_oops",
        task_id=task_id,
        video_id=video_id,
        video_path=Path("/tmp/fake_missing.mp4"),
        instructions=instructions,
        segments=segments,
    )


def test_stride_aligned_completion_at_30_for_end_28() -> None:
    record = fake_record(
        instructions=["step zero", "step one"],
        segments=[
            Segment(start=0.0, end=28.0, instruction_index=0, instruction="step zero", is_mistake=False),
            Segment(start=28.0, end=40.0, instruction_index=1, instruction="step one", is_mistake=False),
        ],
    )
    # Monkeypatch duration via a long second segment and no real video file:
    # build_module_a_windows falls back to max segment end when video is missing/empty.
    windows = build_module_a_windows(
        [record],
        config=ModuleAConfig(
            stride_seconds=5.0,
            completion_margin=0.10,
            negative_to_positive_ratio=100,
            keep_last_wait_windows=100,
            seed=0,
            label_mode="step_id",
        ),
    )
    first_attempt = [window for window in windows if window.step_index == 0]
    assert first_attempt, "expected windows for instruction 0"
    complete = [window for window in first_attempt if window.label != MODULE_A_STEP_ID_NONE_LETTER]
    assert len(complete) == 1
    assert complete[0].window_start == 0.0
    assert complete[0].window_end == 30.0
    assert complete[0].label == "C"
    assert all(window.label == MODULE_A_STEP_ID_NONE_LETTER for window in first_attempt if window is not complete[0])


def test_consecutive_extra_segments_are_separate_b_completions() -> None:
    record = fake_record(
        instructions=["step zero"],
        segments=[
            Segment(
                start=0.0,
                end=10.0,
                instruction_index=-1,
                instruction=EXTRA_STEP_INSTRUCTION,
                is_mistake=True,
            ),
            Segment(
                start=10.0,
                end=20.0,
                instruction_index=-1,
                instruction=EXTRA_STEP_INSTRUCTION,
                is_mistake=True,
            ),
        ],
    )
    windows = build_module_a_windows(
        [record],
        config=ModuleAConfig(
            stride_seconds=5.0,
            completion_margin=0.0,
            negative_to_positive_ratio=100,
            keep_last_wait_windows=100,
            seed=0,
            label_mode="step_id",
        ),
    )
    extras = [window for window in windows if window.label == MODULE_A_STEP_ID_EXTRA_LETTER]
    assert len(extras) == 2
    assert extras[0].window_start == 0.0
    assert extras[0].window_end == 10.0
    assert extras[1].window_start == 10.0
    assert extras[1].window_end == 20.0
    none_before_second = [
        window
        for window in windows
        if window.label == MODULE_A_STEP_ID_NONE_LETTER and window.window_start == 10.0
    ]
    assert none_before_second, "second EXTRA attempt should emit its own NONE windows"


def test_legacy_path_unchanged_wait_complete() -> None:
    record = fake_record(
        instructions=["step zero", "step one"],
        segments=[
            Segment(start=0.0, end=12.0, instruction_index=0, instruction="step zero", is_mistake=False),
            Segment(
                start=12.0,
                end=18.0,
                instruction_index=-1,
                instruction=EXTRA_STEP_INSTRUCTION,
                is_mistake=True,
            ),
            Segment(start=18.0, end=30.0, instruction_index=1, instruction="step one", is_mistake=False),
        ],
    )
    windows = build_module_a_windows(
        [record],
        config=ModuleAConfig(
            stride_seconds=5.0,
            completion_margin=0.0,
            negative_to_positive_ratio=100,
            keep_last_wait_windows=100,
            seed=0,
            label_mode="legacy",
        ),
    )
    assert all(window.label in {"WAIT", "COMPLETE"} for window in windows)
    assert all(window.step_index >= 0 for window in windows)
    completes = [window for window in windows if window.label == "COMPLETE"]
    assert len(completes) == 2


def test_letter_parser_and_instruction_mapping() -> None:
    assert extract_module_a_label("C", label_mode="step_id") == "C"
    assert extract_module_a_label("  b. EXTRA", label_mode="step_id") == "B"
    assert extract_module_a_label("D", label_mode="step_id") == "D"
    assert extract_module_a_label("D\n", label_mode="step_id") == "D"
    assert extract_module_a_label("COMPLETE", label_mode="legacy") == "COMPLETE"
    assert extract_module_a_label("please WAIT", label_mode="legacy") == "WAIT"
    assert module_a_step_id_letter_for_instruction_index(-1) == "B"
    assert module_a_step_id_letter_for_instruction_index(0) == "C"
    assert module_a_step_id_letter_for_instruction_index(1) == "D"
    instructions = ("alpha", "beta")
    assert module_a_instruction_for_step_id_letter("A", instructions) is None
    assert module_a_instruction_for_step_id_letter("B", instructions) == EXTRA_STEP_INSTRUCTION
    assert module_a_instruction_for_step_id_letter("C", instructions) == "alpha"
    assert module_a_instruction_for_step_id_letter("D", instructions) == "beta"
    assert is_module_a_completion_label("A", label_mode="step_id") is False
    assert is_module_a_completion_label("C", label_mode="step_id") is True
    assert is_module_a_completion_label("COMPLETE", label_mode="legacy") is True
    assert is_module_a_completion_label("WAIT", label_mode="legacy") is False


def test_step_id_label_histogram_has_none_procedural_and_extra() -> None:
    record = fake_record(
        instructions=["step zero", "step one"],
        segments=[
            Segment(start=0.0, end=8.0, instruction_index=0, instruction="step zero", is_mistake=False),
            Segment(
                start=8.0,
                end=14.0,
                instruction_index=-1,
                instruction=EXTRA_STEP_INSTRUCTION,
                is_mistake=True,
            ),
            Segment(start=14.0, end=24.0, instruction_index=1, instruction="step one", is_mistake=False),
        ],
    )
    windows = build_module_a_windows(
        [record],
        config=ModuleAConfig(
            stride_seconds=5.0,
            completion_margin=0.0,
            negative_to_positive_ratio=100,
            keep_last_wait_windows=100,
            seed=0,
            label_mode="step_id",
        ),
    )
    counts = Counter(window.label for window in windows)
    assert counts[MODULE_A_STEP_ID_NONE_LETTER] > 0
    assert counts["C"] == 1
    assert counts["D"] == 1
    assert counts[MODULE_A_STEP_ID_EXTRA_LETTER] == 1


def test_multiclass_f1_and_confusion_and_auc() -> None:
    from train_module_a_unsloth import (  # noqa: WPS433
        binary_roc_auc,
        confusion_matrix_from_metrics,
        multiclass_letter_metrics,
    )

    targets = ["A", "A", "C", "C", "B"]
    preds = ["A", "C", "C", "A", "B"]
    metrics = multiclass_letter_metrics(targets, preds, prefix="eval_step_id")
    assert "eval_step_id/macro_f1" in metrics
    assert "eval_step_id/micro_f1" in metrics
    assert "eval_step_id/weighted_f1" in metrics
    assert "eval_step_id/gate/f1" in metrics
    assert metrics["eval_step_id/f1_B"] == 1.0
    matrix = confusion_matrix_from_metrics(metrics, prefix="eval_step_id")
    assert matrix["labels"] == ["A", "B", "C"]
    assert matrix["matrix"][0][0] == 1  # A->A
    assert abs(binary_roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) - 0.75) < 1e-6
    assert binary_roc_auc([1, 1, 1], [0.2, 0.3, 0.9]) is None


def main() -> None:
    test_stride_aligned_completion_at_30_for_end_28()
    test_consecutive_extra_segments_are_separate_b_completions()
    test_legacy_path_unchanged_wait_complete()
    test_letter_parser_and_instruction_mapping()
    test_step_id_label_histogram_has_none_procedural_and_extra()
    test_multiclass_f1_and_confusion_and_auc()
    print("Module A step_id tests passed.")


if __name__ == "__main__":
    main()
