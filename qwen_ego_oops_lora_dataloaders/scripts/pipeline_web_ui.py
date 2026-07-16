from __future__ import annotations

import argparse
import json
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for root in (PROJECT_ROOT, SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from qwen_omd_dataloaders.build import DEFAULT_METADATA_PATH, DEFAULT_MISTAKE_CLASSES_PATH, DEFAULT_VIDEO_ROOT
from qwen_omd_dataloaders.datasets import EgoOopsProvider
from qwen_omd_dataloaders.datasets.ego_oops import _module_b_prompt, _module_c_prompt
from qwen_omd_dataloaders.schema import TrainingExample, VideoRecord, WindowSpec
from qwen_omd_dataloaders.video import video_duration_seconds

from run_online_pipeline_eval import make_collator, make_model_args
from train_module_a_unsloth import (
    ModuleADetectionMetricsCallback,
    extract_module_a_label,
    load_generation_eval_model,
    release_cuda_memory,
    training_example_to_conversation as module_a_to_conversation,
)
from train_module_b_unsloth import (
    ModuleBTemporalMetricsCallback,
    parse_temporal_prediction,
    training_example_to_conversation as module_b_to_conversation,
)
from train_module_c_unsloth import (
    ModuleCMistakeMetricsCallback,
    parse_module_c_prediction,
    training_example_to_conversation as module_c_to_conversation,
)

STATIC_DIR = SCRIPT_ROOT / "pipeline_web_ui_static"


class ClipRequest(BaseModel):
    video_id: str
    step_index: int = Field(ge=0)
    clip_start: float = Field(ge=0.0)
    clip_end: float = Field(gt=0.0)


class ModuleCRequest(ClipRequest):
    predicted_start: float | None = Field(default=None, ge=0.0)
    predicted_end: float | None = Field(default=None, gt=0.0)


class ModelSettings(BaseModel):
    max_frames: int
    vision_resize: int
    max_seq_length: int
    max_new_tokens: int


class AppConfig(BaseModel):
    metadata: str
    mistake_classes: str
    video_root: str
    module_a_checkpoint: str
    module_b_checkpoint: str
    module_c_checkpoint: str
    model_name: str
    load_in_4bit: bool
    load_in_16bit: bool
    max_videos: int
    fps: float
    min_frames: int
    step_seconds: float
    module_a: ModelSettings
    module_b: ModelSettings
    module_c: ModelSettings


class SingleActiveModelManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.active_module: Literal["A", "B", "C"] | None = None
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.lock = threading.Lock()
        self.prediction_lock = threading.Lock()

    def unload(self) -> None:
        with self.lock:
            self._unload_locked()

    def _unload_locked(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        self.tokenizer = None
        self.active_module = None
        release_cuda_memory()

    def load(self, module: Literal["A", "B", "C"]) -> tuple[Any, Any]:
        with self.lock:
            if self.active_module == module and self.model is not None and self.tokenizer is not None:
                return self.model, self.tokenizer
            self._unload_locked()
            checkpoint, settings = self._module_config(module)
            args = argparse.Namespace(
                model_name=self.config.model_name,
                module_a_checkpoint=self.config.module_a_checkpoint,
                module_b_checkpoint=self.config.module_b_checkpoint,
                module_c_checkpoint=self.config.module_c_checkpoint,
                load_in_4bit=self.config.load_in_4bit,
                load_in_16bit=self.config.load_in_16bit,
            )
            self.model, self.tokenizer = load_generation_eval_model(
                make_model_args(args, checkpoint, settings.max_seq_length, settings.vision_resize)
            )
            self.active_module = module
            return self.model, self.tokenizer

    def _module_config(self, module: Literal["A", "B", "C"]) -> tuple[str, ModelSettings]:
        if module == "A":
            return self.config.module_a_checkpoint, self.config.module_a
        if module == "B":
            return self.config.module_b_checkpoint, self.config.module_b
        return self.config.module_c_checkpoint, self.config.module_c


class PipelineWebState:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        provider = EgoOopsProvider(
            metadata_path=config.metadata,
            mistake_classes_path=config.mistake_classes,
            video_root=config.video_root,
            max_videos=config.max_videos,
            require_existing_videos=True,
        )
        self.records = list(provider.iter_video_records())
        self.records_by_id = {record.video_id: record for record in self.records}
        self.model_manager = SingleActiveModelManager(config)

    def get_record(self, video_id: str) -> VideoRecord:
        record = self.records_by_id.get(video_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Unknown video_id: {video_id}")
        return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local UI for stepping through Module A -> B -> C.")
    parser.add_argument("--metadata", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--mistake-classes", default=DEFAULT_MISTAKE_CLASSES_PATH)
    parser.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--module-a-checkpoint", required=True)
    parser.add_argument("--module-b-checkpoint", required=True)
    parser.add_argument("--module-c-checkpoint", required=True)
    parser.add_argument("--model-name", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-16bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-videos", type=int, default=50)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--step-seconds", type=float, default=5.0)
    parser.add_argument("--max-frames-a", type=int, default=16)
    parser.add_argument("--max-frames-b", type=int, default=16)
    parser.add_argument("--max-frames-c", type=int, default=8)
    parser.add_argument("--vision-resize-a", type=int, default=336)
    parser.add_argument("--vision-resize-b", type=int, default=384)
    parser.add_argument("--vision-resize-c", type=int, default=336)
    parser.add_argument("--max-seq-length-a", type=int, default=3072)
    parser.add_argument("--max-seq-length-b", type=int, default=5120)
    parser.add_argument("--max-seq-length-c", type=int, default=3072)
    parser.add_argument("--module-a-max-new-tokens", type=int, default=8)
    parser.add_argument("--module-b-max-new-tokens", type=int, default=64)
    parser.add_argument("--module-c-max-new-tokens", type=int, default=128)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> AppConfig:
    return AppConfig(
        metadata=str(args.metadata),
        mistake_classes=str(args.mistake_classes),
        video_root=str(args.video_root),
        module_a_checkpoint=str(args.module_a_checkpoint),
        module_b_checkpoint=str(args.module_b_checkpoint),
        module_c_checkpoint=str(args.module_c_checkpoint),
        model_name=str(args.model_name),
        load_in_4bit=bool(args.load_in_4bit),
        load_in_16bit=bool(args.load_in_16bit),
        max_videos=int(args.max_videos),
        fps=float(args.fps),
        min_frames=int(args.min_frames),
        step_seconds=float(args.step_seconds),
        module_a=ModelSettings(
            max_frames=int(args.max_frames_a),
            vision_resize=int(args.vision_resize_a),
            max_seq_length=int(args.max_seq_length_a),
            max_new_tokens=int(args.module_a_max_new_tokens),
        ),
        module_b=ModelSettings(
            max_frames=int(args.max_frames_b),
            vision_resize=int(args.vision_resize_b),
            max_seq_length=int(args.max_seq_length_b),
            max_new_tokens=int(args.module_b_max_new_tokens),
        ),
        module_c=ModelSettings(
            max_frames=int(args.max_frames_c),
            vision_resize=int(args.vision_resize_c),
            max_seq_length=int(args.max_seq_length_c),
            max_new_tokens=int(args.module_c_max_new_tokens),
        ),
    )


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="Online Mistake Detection Pipeline UI")
    state = PipelineWebState(config)
    app.state.pipeline = state

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        return {"config": model_to_dict(config)}

    @app.get("/api/videos")
    def list_videos() -> dict[str, Any]:
        videos = [video_summary(record, config.step_seconds) for record in state.records]
        return {"videos": videos}

    @app.get("/api/videos/{video_id}")
    def get_video(video_id: str) -> dict[str, Any]:
        record = state.get_record(video_id)
        return {"video": video_detail(record, config.step_seconds)}

    @app.get("/api/video-file/{video_id}")
    def get_video_file(video_id: str) -> FileResponse:
        record = state.get_record(video_id)
        return FileResponse(record.video_path, media_type="video/mp4", filename=record.video_path.name)

    @app.post("/api/module-a")
    def run_module_a(request: ClipRequest) -> dict[str, Any]:
        with state.model_manager.prediction_lock:
            record = state.get_record(request.video_id)
            window = request_to_window(record, request)
            model, tokenizer = state.model_manager.load("A")
            settings = config.module_a
            collator = make_collator(model, tokenizer, settings.max_seq_length, settings.vision_resize)
            generator = ModuleADetectionMetricsCallback(
                eval_examples=[],
                data_collator=collator,
                tokenizer=tokenizer,
                max_samples=-1,
                max_new_tokens=settings.max_new_tokens,
                beta=2.0,
                label_score_eval=False,
            )
            example = module_a_to_conversation(
                window_to_example(window),
                fps=config.fps,
                min_frames=config.min_frames,
                max_frames=settings.max_frames,
            )
            raw = generator._generate_label(model, example)
            prediction = extract_module_a_label(raw)
            return {"module": "A", "raw": raw, "prediction": prediction, "window": window_payload(window)}

    @app.post("/api/module-b")
    def run_module_b(request: ClipRequest) -> dict[str, Any]:
        with state.model_manager.prediction_lock:
            record = state.get_record(request.video_id)
            window = request_to_window(record, request)
            model, tokenizer = state.model_manager.load("B")
            settings = config.module_b
            collator = make_collator(model, tokenizer, settings.max_seq_length, settings.vision_resize)
            generator = ModuleBTemporalMetricsCallback(
                eval_examples=[],
                data_collator=collator,
                tokenizer=tokenizer,
                max_samples=-1,
                max_new_tokens=settings.max_new_tokens,
            )
            example = module_b_example(window)
            conversation = module_b_to_conversation(
                example,
                fps=config.fps,
                min_frames=config.min_frames,
                max_frames=settings.max_frames,
            )
            raw = generator._generate_span(model, conversation)
            parsed = parse_temporal_prediction(raw)
            global_span = None
            if parsed is not None:
                global_span = [window.window_start + parsed[0], window.window_start + parsed[1]]
            return {
                "module": "B",
                "raw": raw,
                "prediction": None if parsed is None else list(parsed),
                "global_prediction": global_span,
                "window": window_payload(window),
            }

    @app.post("/api/module-c")
    def run_module_c(request: ModuleCRequest) -> dict[str, Any]:
        with state.model_manager.prediction_lock:
            record = state.get_record(request.video_id)
            clip_start = request.predicted_start if request.predicted_start is not None else request.clip_start
            clip_end = request.predicted_end if request.predicted_end is not None else request.clip_end
            c_request = ClipRequest(
                video_id=request.video_id,
                step_index=request.step_index,
                clip_start=float(clip_start),
                clip_end=float(clip_end),
            )
            window = request_to_window(record, c_request)
            model, tokenizer = state.model_manager.load("C")
            settings = config.module_c
            collator = make_collator(model, tokenizer, settings.max_seq_length, settings.vision_resize)
            generator = ModuleCMistakeMetricsCallback(
                eval_examples=[],
                data_collator=collator,
                tokenizer=tokenizer,
                max_samples=-1,
                max_new_tokens=settings.max_new_tokens,
            )
            conversation = module_c_to_conversation(
                module_c_example(window),
                fps=config.fps,
                min_frames=config.min_frames,
                max_frames=settings.max_frames,
            )
            raw = generator._generate_json(model, conversation)
            parsed = parse_module_c_prediction(raw)
            return {"module": "C", "raw": raw, "prediction": parsed, "window": window_payload(window)}

    @app.post("/api/unload-model")
    def unload_model() -> dict[str, Any]:
        state.model_manager.unload()
        return {"status": "unloaded"}

    return app


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def video_summary(record: VideoRecord, step_seconds: float) -> dict[str, Any]:
    duration = video_duration_seconds(record.video_path)
    return {
        "video_id": record.video_id,
        "task_id": record.task_id,
        "duration": duration,
        "num_instructions": len(record.instructions),
        "video_url": f"/api/video-file/{record.video_id}",
        "step_seconds": step_seconds,
    }


def video_detail(record: VideoRecord, step_seconds: float) -> dict[str, Any]:
    payload = video_summary(record, step_seconds)
    payload["instructions"] = [
        {"step_index": index, "text": instruction}
        for index, instruction in enumerate(record.instructions)
    ]
    payload["segments"] = [
        {
            "start": segment.start,
            "end": segment.end,
            "instruction_index": segment.instruction_index,
            "instruction": segment.instruction,
            "is_mistake": segment.is_mistake,
            "error_label": segment.error_label,
            "caption": segment.caption,
        }
        for segment in record.segments
    ]
    return payload


def request_to_window(record: VideoRecord, request: ClipRequest) -> WindowSpec:
    if request.clip_end <= request.clip_start:
        raise HTTPException(status_code=400, detail="clip_end must be greater than clip_start.")
    if request.step_index >= len(record.instructions):
        raise HTTPException(status_code=400, detail=f"step_index out of range: {request.step_index}")
    gt_start, gt_end = nearest_segment_bounds(record, request.step_index)
    return WindowSpec(
        video_path=record.video_path,
        video_id=record.video_id,
        task_id=record.task_id,
        step_index=request.step_index,
        current_step=record.instructions[request.step_index],
        window_start=float(request.clip_start),
        window_end=float(request.clip_end),
        gt_start=gt_start,
        gt_end=gt_end,
        label="WAIT",
        completed_steps=tuple(record.instructions[: request.step_index]),
        pending_steps=tuple(record.instructions[request.step_index + 1 :]),
    )


def nearest_segment_bounds(record: VideoRecord, step_index: int) -> tuple[float | None, float | None]:
    candidates = [segment for segment in record.segments if segment.instruction_index == step_index]
    if not candidates:
        return None, None
    segment = candidates[0]
    return segment.start, segment.end


def window_to_example(window: WindowSpec) -> TrainingExample:
    prompt = (
        "Completed Steps: "
        f"{json.dumps(list(window.completed_steps), ensure_ascii=False)}\n"
        f"Current Step: {window.current_step}\n"
        "Pending Steps: "
        f"{json.dumps(list(window.pending_steps), ensure_ascii=False)}\n"
        "Status?"
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
        prompt_text=prompt,
        target_text="WAIT",
        gt_start=window.gt_start,
        gt_end=window.gt_end,
        label=window.label,
    )


def module_b_example(window: WindowSpec) -> TrainingExample:
    return TrainingExample(
        module="B",
        source_dataset="ego_oops",
        video_path=window.video_path,
        video_id=window.video_id,
        task_id=window.task_id,
        step_index=window.step_index,
        window_start=window.window_start,
        window_end=window.window_end,
        prompt_text=_module_b_prompt(window.current_step),
        target_text="not completed",
        gt_start=window.gt_start,
        gt_end=window.gt_end,
        label="LOCALIZE",
        metadata={"instruction": window.current_step},
    )


def module_c_example(window: WindowSpec) -> TrainingExample:
    return TrainingExample(
        module="C",
        source_dataset="ego_oops",
        video_path=window.video_path,
        video_id=window.video_id,
        task_id=window.task_id,
        step_index=window.step_index,
        window_start=window.window_start,
        window_end=window.window_end,
        prompt_text=_module_c_prompt(window.current_step),
        target_text='{"mistake":false,"reasoning":""}',
        gt_start=window.gt_start,
        gt_end=window.gt_end,
        metadata={"instruction": window.current_step},
    )


def window_payload(window: WindowSpec) -> dict[str, Any]:
    payload = asdict(window)
    payload["video_path"] = str(window.video_path)
    return payload


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    app = create_app(config)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
