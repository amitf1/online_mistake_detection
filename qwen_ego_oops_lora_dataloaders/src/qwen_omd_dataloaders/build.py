from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ModuleAConfig, ModuleBConfig, ModuleCConfig, TimeTokenConfig, VideoSamplingConfig
from .datasets import EgoOopsModuleADataset, EgoOopsModuleBDataset, EgoOopsModuleCDataset, EgoOopsProvider


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METADATA_PATH = REPO_ROOT / "ego_oops" / "EgoOops-annotations" / "meta" / "metadata_edited.json"
DEFAULT_MISTAKE_CLASSES_PATH = REPO_ROOT / "ego_oops" / "EgoOops-annotations" / "meta" / "mistake_classes.json"
DEFAULT_VIDEO_ROOT = Path("/nvcr/users/afeldman/data/exper/videos-processed-720p")


def build_processor(
    model_id: str = "Qwen/Qwen3.5-4B",
    *,
    trust_remote_code: bool = True,
    add_time_tokens: bool = True,
    time_config: TimeTokenConfig | None = None,
) -> Any:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if add_time_tokens:
        add_special_training_tokens(processor, time_config=time_config)
    return processor


def add_special_training_tokens(
    processor: Any,
    *,
    time_config: TimeTokenConfig | None = None,
) -> int:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return 0
    cfg = time_config or TimeTokenConfig()
    tokens = [
        cfg.token_template.format(index=index)
        for index in range(cfg.num_bins)
    ]
    tokens.extend([cfg.no_action_token, "[WAIT]", "[COMPLETE]"])
    existing_vocab = tokenizer.get_vocab()
    new_tokens = [token for token in tokens if token not in existing_vocab]
    if not new_tokens:
        return 0
    return int(tokenizer.add_special_tokens({"additional_special_tokens": new_tokens}))


def build_ego_oops_provider(
    *,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    mistake_classes_path: str | Path = DEFAULT_MISTAKE_CLASSES_PATH,
    video_root: str | Path = DEFAULT_VIDEO_ROOT,
    video_ids: set[str] | None = None,
    task_ids: set[str] | None = None,
    max_videos: int | None = 50,
    require_existing_videos: bool = True,
) -> EgoOopsProvider:
    return EgoOopsProvider(
        metadata_path=metadata_path,
        mistake_classes_path=mistake_classes_path,
        video_root=video_root,
        video_ids=video_ids,
        task_ids=task_ids,
        max_videos=max_videos,
        require_existing_videos=require_existing_videos,
    )


def build_datasets(
    *,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    mistake_classes_path: str | Path = DEFAULT_MISTAKE_CLASSES_PATH,
    video_root: str | Path = DEFAULT_VIDEO_ROOT,
    video_ids: set[str] | None = None,
    task_ids: set[str] | None = None,
    max_videos: int | None = 50,
    module_a_config: ModuleAConfig | None = None,
    module_b_config: ModuleBConfig | None = None,
    module_c_config: ModuleCConfig | None = None,
    time_config: TimeTokenConfig | None = None,
) -> dict[str, Any]:
    provider_kwargs = dict(
        metadata_path=metadata_path,
        mistake_classes_path=mistake_classes_path,
        video_root=video_root,
        video_ids=video_ids,
        task_ids=task_ids,
        max_videos=max_videos,
    )
    return {
        "A": EgoOopsModuleADataset(
            build_ego_oops_provider(**provider_kwargs),
            config=module_a_config,
        ),
        "B": EgoOopsModuleBDataset(
            build_ego_oops_provider(**provider_kwargs),
            module_a_config=module_a_config,
            config=module_b_config,
            time_config=time_config,
        ),
        "C": EgoOopsModuleCDataset(
            build_ego_oops_provider(**provider_kwargs),
            config=module_c_config,
        ),
    }


def build_dataloaders(
    *,
    processor: Any,
    datasets: dict[str, Any] | None = None,
    video_config_a: VideoSamplingConfig | None = None,
    video_config_b: VideoSamplingConfig | None = None,
    video_config_c: VideoSamplingConfig | None = None,
    batch_size: int = 1,
    num_workers: int = 2,
    pin_memory: bool = True,
    include_metadata: bool = False,
    **dataset_kwargs: Any,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    from .collators import ModuleACollator, ModuleBCollator, ModuleCCollator

    datasets = datasets or build_datasets(**dataset_kwargs)
    persistent_workers = num_workers > 0
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return {
        "A": DataLoader(
            datasets["A"],
            shuffle=True,
            collate_fn=ModuleACollator(
                processor=processor,
                video_config=video_config_a or VideoSamplingConfig(max_frames=64),
                include_metadata=include_metadata,
            ),
            **common,
        ),
        "B": DataLoader(
            datasets["B"],
            shuffle=True,
            collate_fn=ModuleBCollator(
                processor=processor,
                video_config=video_config_b or VideoSamplingConfig(max_frames=64),
                include_metadata=include_metadata,
            ),
            **common,
        ),
        "C": DataLoader(
            datasets["C"],
            shuffle=True,
            collate_fn=ModuleCCollator(
                processor=processor,
                video_config=video_config_c or VideoSamplingConfig(max_frames=48),
                include_metadata=include_metadata,
            ),
            **common,
        ),
    }
