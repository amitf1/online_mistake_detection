from __future__ import annotations

from typing import Iterable

from ..schema import VideoRecord


class _NotImplementedProvider:
    dataset_name = "unknown"
    expected_contract = (
        "Provider must implement iter_video_records() and yield VideoRecord "
        "objects with normalized instructions and Segment entries."
    )

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def iter_video_records(self) -> Iterable[VideoRecord]:
        raise NotImplementedError(f"{self.dataset_name}: {self.expected_contract}")


class MattBenchProvider(_NotImplementedProvider):
    dataset_name = "MATT-Bench / Ego4D-M"
    expected_contract = (
        "Map Semantic Role Labeling mismatches into Segment.error_label and "
        "synthesize reasoning from predicate/argument mismatch metadata."
    )


class Assembly101OProvider(_NotImplementedProvider):
    dataset_name = "Assembly101-O"
    expected_contract = (
        "Map assembly error annotations into normalized segments; synthesize "
        "reasoning from the available object/action/order error class."
    )


class EpicTentOProvider(_NotImplementedProvider):
    dataset_name = "Epic-Tent-O"
    expected_contract = (
        "Map tent assembly errors into normalized segments; synthesize reasoning "
        "from action/object/order labels when free-text explanations are absent."
    )


class EgoPERProvider(_NotImplementedProvider):
    dataset_name = "EgoPER"
    expected_contract = (
        "Map scripted cooking activities and scripted errors into normalized "
        "VideoRecord/Segment objects across the five cooking activities."
    )
