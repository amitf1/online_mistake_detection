from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import qwen_omd_dataloaders.datasets.ego_oops as ego_oops  # noqa: E402
from qwen_omd_dataloaders.config import ModuleCConfig  # noqa: E402
from qwen_omd_dataloaders.datasets import EgoOopsModuleCDataset  # noqa: E402
from qwen_omd_dataloaders.schema import Segment, VideoRecord  # noqa: E402
from train_module_c_unsloth import parse_module_c_prediction, module_c_metrics  # noqa: E402


class FakeProvider:
    def __init__(self, records: list[VideoRecord]) -> None:
        self.records = records

    def iter_video_records(self) -> list[VideoRecord]:
        return self.records


def build_record() -> VideoRecord:
    return VideoRecord(
        dataset="ego_oops",
        task_id="task",
        video_id="video",
        video_path=Path("/tmp/fake.mp4"),
        instructions=["pick up the red block", "place it on the table"],
        segments=[
            Segment(
                start=1.0,
                end=3.0,
                instruction_index=0,
                instruction="pick up the red block",
                is_mistake=False,
                raw={"labels": []},
            ),
            Segment(
                start=4.0,
                end=6.0,
                instruction_index=1,
                instruction="place it on the table",
                is_mistake=True,
                error_label="working with wrong objects",
                caption="The person places the wrong object on the table.",
                raw={"labels": [0]},
            ),
            Segment(
                start=7.0,
                end=8.0,
                instruction_index=-1,
                instruction="undefined / extra mistake action",
                is_mistake=True,
                error_label="others",
                caption="",
                raw={"labels": [5]},
            ),
        ],
    )


def build_dataset() -> EgoOopsModuleCDataset:
    old_duration = ego_oops.video_duration_seconds
    ego_oops.video_duration_seconds = lambda _path: 10.0
    try:
        return EgoOopsModuleCDataset(
            FakeProvider([build_record()]),
            config=ModuleCConfig(
                min_duration_seconds=0.5,
                jitter_ratio_min=0.05,
                jitter_ratio_max=0.10,
                seed=123,
            ),
        )
    finally:
        ego_oops.video_duration_seconds = old_duration


def test_module_c_target_json_and_reasoning() -> None:
    dataset = build_dataset()
    assert len(dataset) == 3

    correct = dataset[0]
    correct_target = json.loads(correct.target_text)
    assert list(correct_target) == ["mistake", "reasoning"]
    assert correct_target["mistake"] is False
    assert correct_target["reasoning"] == (
        "The observed action follows the instructed step without a visible procedural deviation."
    )
    assert correct.label == "CORRECT"

    mistake = dataset[1]
    mistake_target = json.loads(mistake.target_text)
    assert mistake_target == {
        "mistake": True,
        "reasoning": "The person places the wrong object on the table.",
    }
    assert mistake.metadata["mistake_labels"] == ["working with wrong objects"]
    assert mistake.metadata["raw_labels"] == [0]

    fallback = dataset[2]
    fallback_target = json.loads(fallback.target_text)
    assert fallback_target["mistake"] is True
    assert fallback_target["reasoning"] == "The execution deviates from the instruction and is labeled as others."
    assert fallback.metadata["is_undefined_instruction"] is True


def test_module_c_prompt_mentions_mistake_and_json() -> None:
    dataset = build_dataset()
    prompt = dataset[0].prompt_text
    assert "EgoOops-style mistake" in prompt
    assert "ordinary execution variation" in prompt
    assert "Return JSON only" in prompt


def test_module_c_jitter_keeps_valid_windows() -> None:
    dataset = build_dataset()
    for example in dataset:
        assert 0.0 <= example.window_start < example.window_end <= 10.0
        assert example.window_end - example.window_start >= 0.5
        assert example.gt_start is not None
        assert example.gt_end is not None
        assert example.window_start <= example.gt_end
        assert example.window_end >= example.gt_start


def test_parse_module_c_prediction() -> None:
    assert parse_module_c_prediction('{"mistake":true,"reasoning":"wrong object"}') == {
        "mistake": True,
        "reasoning": "wrong object",
    }
    assert parse_module_c_prediction('```json\n{"mistake":"false","reasoning":"follows the instruction"}\n```') == {
        "mistake": False,
        "reasoning": "follows the instruction",
    }
    assert parse_module_c_prediction('prefix {"mistake":"mistake","reasoning":"self correction"} suffix') == {
        "mistake": True,
        "reasoning": "self correction",
    }
    assert parse_module_c_prediction('{"mistake":true}') is None
    assert parse_module_c_prediction("not json") is None


def test_module_c_metrics() -> None:
    metrics = module_c_metrics(
        [True, True, False, False],
        [True, None, True, False],
        target_reasonings=["wrong object", "unintended action", "correct action", "correct action"],
        predicted_reasonings=["wrong object", None, "wrong color", "correct action"],
    )
    assert metrics["eval_mistake/num_samples"] == 4.0
    assert metrics["eval_mistake/tp"] == 1.0
    assert metrics["eval_mistake/fp"] == 1.0
    assert metrics["eval_mistake/tn"] == 1.0
    assert metrics["eval_mistake/fn"] == 1.0
    assert metrics["eval_mistake/invalid_json_rate"] == 0.25
    assert metrics["eval_mistake/f1"] == 0.5
    assert metrics["eval_reasoning/nonempty_rate"] == 0.75
    assert 0.0 <= metrics["eval_reasoning/lexical_overlap"] <= 1.0


def main() -> None:
    test_module_c_target_json_and_reasoning()
    test_module_c_prompt_mentions_mistake_and_json()
    test_module_c_jitter_keeps_valid_windows()
    test_parse_module_c_prediction()
    test_module_c_metrics()
    print("Module C reasoning tests passed.")


if __name__ == "__main__":
    main()
