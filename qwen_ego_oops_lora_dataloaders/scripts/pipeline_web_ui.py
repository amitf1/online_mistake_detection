from __future__ import annotations

import argparse
import sys
import threading
import time
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
from qwen_omd_dataloaders.datasets.ego_oops import (
    EXTRA_STEP_INSTRUCTION,
    MODULE_A_STEP_ID_EXTRA_LETTER,
    MODULE_A_STEP_ID_NONE_LETTER,
    _module_a_prompt,
    _module_a_step_id_prompt,
    _module_b_prompt,
    _module_c_prompt,
    is_module_a_completion_label,
    module_a_instruction_for_step_id_letter,
    module_a_step_id_letter_for_instruction_index,
)
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
    step_index: int = Field(ge=-1)
    clip_start: float = Field(ge=0.0)
    clip_end: float = Field(gt=0.0)
    module_a_prediction: str | None = None
    completed_steps: list[str] | None = None
    next_clip_start_from_b: bool = False
    predicted_global_end: float | None = Field(default=None, ge=0.0)


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
    module_a_label_mode: str
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
        self.mistake_classes = list(provider.mistake_classes)
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
    parser.add_argument(
        "--module-a-label-mode",
        choices=("legacy", "step_id"),
        default="step_id",
        help="Module A prompt/parse mode for the UI.",
    )
    parser.add_argument("--max-frames-a", type=int, default=16)
    parser.add_argument("--max-frames-b", type=int, default=16)
    parser.add_argument("--max-frames-c", type=int, default=16)
    parser.add_argument("--vision-resize-a", type=int, default=336)
    parser.add_argument("--vision-resize-b", type=int, default=384)
    parser.add_argument("--vision-resize-c", type=int, default=384)
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
        module_a_label_mode=str(args.module_a_label_mode),
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
        request_started = time.perf_counter()
        with state.model_manager.prediction_lock:
            record = state.get_record(request.video_id)
            window = request_to_window(record, request, label_mode=config.module_a_label_mode)
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
                window_to_example(window, mistake_classes=state.mistake_classes),
                fps=config.fps,
                min_frames=config.min_frames,
                max_frames=settings.max_frames,
            )
            prediction_started = time.perf_counter()
            raw = generator._generate_label(model, example)
            prediction_seconds = time.perf_counter() - prediction_started
            prediction = extract_module_a_label(raw, label_mode=config.module_a_label_mode)
            resolved = module_a_instruction_for_step_id_letter(
                prediction,
                window.all_instructions or tuple(record.instructions),
            )
            return {
                "module": "A",
                "raw": raw,
                "prediction": prediction,
                "is_completion": is_module_a_completion_label(
                    prediction,
                    label_mode=config.module_a_label_mode,
                ),
                "resolved_instruction": resolved,
                "prediction_seconds": prediction_seconds,
                "request_wall_seconds": time.perf_counter() - request_started,
                "window": window_payload(window),
            }

    @app.post("/api/module-b")
    def run_module_b(request: ClipRequest) -> dict[str, Any]:
        request_started = time.perf_counter()
        with state.model_manager.prediction_lock:
            record = state.get_record(request.video_id)
            window = request_to_window(record, request, label_mode=config.module_a_label_mode)
            instruction = resolve_instruction_for_request(record, request, window)
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
            example = module_b_example(window, instruction=instruction)
            conversation = module_b_to_conversation(
                example,
                fps=config.fps,
                min_frames=config.min_frames,
                max_frames=settings.max_frames,
            )
            prediction_started = time.perf_counter()
            raw = generator._generate_span(model, conversation)
            prediction_seconds = time.perf_counter() - prediction_started
            parsed = parse_temporal_prediction(raw)
            global_span = None
            next_clip_start = None
            if parsed is not None:
                global_span = [window.window_start + parsed[0], window.window_start + parsed[1]]
                if request.next_clip_start_from_b or config.module_a_label_mode == "step_id":
                    next_clip_start = global_span[1]
            return {
                "module": "B",
                "raw": raw,
                "prediction": None if parsed is None else list(parsed),
                "global_prediction": global_span,
                "instruction": instruction,
                "next_clip_start": next_clip_start,
                "prediction_seconds": prediction_seconds,
                "request_wall_seconds": time.perf_counter() - request_started,
                "window": window_payload(window),
            }

    @app.post("/api/module-c")
    def run_module_c(request: ModuleCRequest) -> dict[str, Any]:
        request_started = time.perf_counter()
        with state.model_manager.prediction_lock:
            record = state.get_record(request.video_id)
            clip_start = request.predicted_start if request.predicted_start is not None else request.clip_start
            clip_end = request.predicted_end if request.predicted_end is not None else request.clip_end
            c_request = ClipRequest(
                video_id=request.video_id,
                step_index=request.step_index,
                clip_start=float(clip_start),
                clip_end=float(clip_end),
                module_a_prediction=request.module_a_prediction,
                completed_steps=request.completed_steps,
            )
            window = request_to_window(record, c_request, label_mode=config.module_a_label_mode)
            instruction = resolve_instruction_for_request(record, c_request, window)
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
                module_c_example(window, instruction=instruction),
                fps=config.fps,
                min_frames=config.min_frames,
                max_frames=settings.max_frames,
            )
            prediction_started = time.perf_counter()
            raw = generator._generate_json(model, conversation)
            prediction_seconds = time.perf_counter() - prediction_started
            parsed = parse_module_c_prediction(raw)
            return {
                "module": "C",
                "raw": raw,
                "prediction": parsed,
                "instruction": instruction,
                "prediction_seconds": prediction_seconds,
                "request_wall_seconds": time.perf_counter() - request_started,
                "window": window_payload(window),
            }

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


def request_to_window(
    record: VideoRecord,
    request: ClipRequest,
    *,
    label_mode: str = "step_id",
) -> WindowSpec:
    if request.clip_end <= request.clip_start:
        raise HTTPException(status_code=400, detail="clip_end must be greater than clip_start.")
    if request.step_index >= len(record.instructions):
        raise HTTPException(status_code=400, detail=f"step_index out of range: {request.step_index}")

    clip_start = float(request.clip_start)
    if request.next_clip_start_from_b and request.predicted_global_end is not None:
        clip_start = float(request.predicted_global_end)

    gt_start, gt_end = nearest_segment_bounds(record, request.step_index)
    all_instructions = tuple(record.instructions)
    if label_mode == "step_id":
        if request.step_index < 0:
            current_step = EXTRA_STEP_INSTRUCTION
            label = (
                MODULE_A_STEP_ID_EXTRA_LETTER
                if gt_end is not None and float(request.clip_end) >= gt_end
                else MODULE_A_STEP_ID_NONE_LETTER
            )
        else:
            current_step = record.instructions[request.step_index]
            complete_letter = module_a_step_id_letter_for_instruction_index(request.step_index)
            label = (
                complete_letter
                if gt_end is not None and float(request.clip_end) >= gt_end
                else MODULE_A_STEP_ID_NONE_LETTER
            )
        if request.completed_steps is not None:
            completed_steps = tuple(request.completed_steps)
        elif request.step_index < 0:
            completed_steps = tuple(
                segment.instruction
                for segment in record.segments
                if segment.instruction_index >= 0 and segment.end <= clip_start + 1e-6
            )
        else:
            completed_steps = tuple(record.instructions[: request.step_index])
        pending_steps = tuple(
            instruction
            for index, instruction in enumerate(record.instructions)
            if index != request.step_index and instruction not in completed_steps
        )
    else:
        if request.step_index < 0:
            raise HTTPException(status_code=400, detail="legacy mode requires step_index >= 0")
        current_step = record.instructions[request.step_index]
        label = "UNKNOWN" if gt_end is None else ("COMPLETE" if float(request.clip_end) >= gt_end else "WAIT")
        completed_steps = tuple(record.instructions[: request.step_index])
        pending_steps = tuple(record.instructions[request.step_index + 1 :])

    return WindowSpec(
        video_path=record.video_path,
        video_id=record.video_id,
        task_id=record.task_id,
        step_index=request.step_index,
        current_step=current_step,
        window_start=clip_start,
        window_end=float(request.clip_end),
        gt_start=gt_start,
        gt_end=gt_end,
        label=label,
        completed_steps=completed_steps,
        pending_steps=pending_steps,
        all_instructions=all_instructions,
        label_mode=label_mode,
    )


def nearest_segment_bounds(record: VideoRecord, step_index: int) -> tuple[float | None, float | None]:
    candidates = [segment for segment in record.segments if segment.instruction_index == step_index]
    if not candidates:
        return None, None
    segment = candidates[0]
    return segment.start, segment.end


def resolve_instruction_for_request(
    record: VideoRecord,
    request: ClipRequest,
    window: WindowSpec,
) -> str:
    if request.module_a_prediction:
        mapped = module_a_instruction_for_step_id_letter(
            request.module_a_prediction,
            window.all_instructions or tuple(record.instructions),
        )
        if mapped is not None:
            return mapped
        if request.module_a_prediction.strip().upper().startswith(MODULE_A_STEP_ID_EXTRA_LETTER):
            return EXTRA_STEP_INSTRUCTION
    if window.step_index < 0:
        return EXTRA_STEP_INSTRUCTION
    return window.current_step


def window_to_example(window: WindowSpec, *, mistake_classes: list[str]) -> TrainingExample:
    if window.label_mode == "step_id":
        prompt = _module_a_step_id_prompt(
            task_id=window.task_id,
            instructions=window.all_instructions or (),
            completed_steps=window.completed_steps,
            mistake_classes=mistake_classes,
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
        prompt_text=prompt,
        target_text=window.label,
        gt_start=window.gt_start,
        gt_end=window.gt_end,
        label=window.label,
        metadata={
            "label_mode": window.label_mode,
            "all_instructions": list(window.all_instructions),
            "completed_steps": list(window.completed_steps),
            "mistake_classes": list(mistake_classes),
        },
    )


def module_b_example(window: WindowSpec, *, instruction: str | None = None) -> TrainingExample:
    resolved = instruction or window.current_step
    return TrainingExample(
        module="B",
        source_dataset="ego_oops",
        video_path=window.video_path,
        video_id=window.video_id,
        task_id=window.task_id,
        step_index=window.step_index,
        window_start=window.window_start,
        window_end=window.window_end,
        prompt_text=_module_b_prompt(resolved),
        target_text="not completed",
        gt_start=window.gt_start,
        gt_end=window.gt_end,
        label="LOCALIZE",
        metadata={"instruction": resolved},
    )


def module_c_example(window: WindowSpec, *, instruction: str | None = None) -> TrainingExample:
    resolved = instruction or window.current_step
    return TrainingExample(
        module="C",
        source_dataset="ego_oops",
        video_path=window.video_path,
        video_id=window.video_id,
        task_id=window.task_id,
        step_index=window.step_index,
        window_start=window.window_start,
        window_end=window.window_end,
        prompt_text=_module_c_prompt(resolved),
        target_text='{"mistake":false,"reasoning":""}',
        gt_start=window.gt_start,
        gt_end=window.gt_end,
        metadata={"instruction": resolved},
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
