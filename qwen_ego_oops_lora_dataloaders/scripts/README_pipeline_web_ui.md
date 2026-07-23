# Local Pipeline Web UI

`pipeline_web_ui.py` serves a local browser UI for manually stepping through the full online pipeline:

```text
selected video clip -> Module A WAIT/COMPLETE -> Module B temporal window -> Module C mistake/reasoning
```

The UI is designed for the local 16GB GPU machine. It keeps only one Qwen/PEFT checkpoint loaded at a time, so switching from Module A to B or C may be slow, but each result is genuine model inference.

## Install Dependencies

Inside the existing training environment or Docker image:

```bash
pip install -r requirements.txt
```

The web UI adds `fastapi` and `uvicorn` to the existing requirements.

## Run Locally On 16GB

From:

```bash
cd /home/amit/online_mistake_detection/online_mistake_detection/qwen_ego_oops_lora_dataloaders
```

Start the server:

```bash
python3 scripts/pipeline_web_ui.py \
  --module-a-checkpoint /path/to/module_a_checkpoint \
  --module-b-checkpoint /path/to/module_b_checkpoint \
  --module-c-checkpoint /path/to/module_c_checkpoint \
  --video-root /home/amit/online_mistake_detection/data/videos-processed-720p \
  --max-videos 50 \
  --max-frames-a 16 \
  --max-frames-b 16 \
  --max-frames-c 16 \
  --vision-resize-a 336 \
  --vision-resize-b 384 \
  --vision-resize-c 384 \
  --max-seq-length-a 3072 \
  --max-seq-length-b 5120 \
  --max-seq-length-c 3072 \
  --host 127.0.0.1 \
  --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

If you run inside Docker, publish the port and mount the checkpoint/video paths into the container.

## Run With Docker

From the same project directory:

```bash
bash scripts/run_pipeline_web_ui_docker.sh
```

The wrapper publishes port `7860`, mounts the video data, outputs, Hugging Face cache, and EgoOops annotations, and defaults to the locally downloaded best checkpoints:

```text
MODULE_A_CHECKPOINT=/home/amit/online_mistake_detection/outputs/module_a_qwen35_2b_lora_wait_complete_vision/runs/module_a_recall_loss2_from_ep8_to_ep50_v8/best_ep10
MODULE_B_CHECKPOINT=/home/amit/online_mistake_detection/outputs/wandb_artifacts/module-b-9kahxivi-best_v1/best_ep7
MODULE_C_CHECKPOINT=/home/amit/online_mistake_detection/outputs/module_c_qwen35_lora_reasoning/runs/module_c_16gb_2b_r16_16f384_seq3072/best_ep1
```

Override any of them if needed:

```bash
PORT=7861 \
MODULE_B_CHECKPOINT=/path/to/other/module_b_checkpoint \
bash scripts/run_pipeline_web_ui_docker.sh
```

Then open:

```text
http://127.0.0.1:7860
```

If the image was built before the web dependencies were added, the wrapper installs `fastapi` and `uvicorn` at startup by default. Set `INSTALL_WEB_DEPS=false` after rebuilding the Docker image with the updated `requirements.txt`.

## Usage Flow

1. Select a video.
2. Select the current instruction/step.
3. Move the start/end sliders. They snap to the configured `--step-seconds`, default `5`.
4. Click `Play Selected Clip`; playback starts at the selected start time and stops at the selected end time.
5. Click `Run Module A` to get `WAIT` or `COMPLETE`.
6. Click `Run Module B` to get the localized attempted-step window. The UI overlays this prediction in green on the timeline.
7. Click `Run Module C` to classify the Module B predicted crop as mistake/correct and show reasoning.
8. Click `Unload GPU Model` if you want to explicitly free GPU memory.

Each result panel shows:

- `Prediction time`: time spent in the actual model prediction call for that module.
- `Request wall time`: total click-to-response time, including checkpoint load/switch overhead if that module was not already active.

## Memory Behavior

The server uses a single-active-model manager:

- First `Run Module A` loads the Module A checkpoint.
- First `Run Module B` unloads Module A, frees CUDA memory, then loads Module B.
- First `Run Module C` unloads Module B, frees CUDA memory, then loads Module C.
- Repeated clicks on the same module reuse the currently loaded model.

This is slower than keeping all models loaded but is much more appropriate for a 16GB GPU.

## API Endpoints

- `GET /api/videos`: list available videos.
- `GET /api/videos/{video_id}`: video metadata, instructions, and annotations.
- `GET /api/video-file/{video_id}`: serves the MP4 file to the browser.
- `POST /api/module-a`: run Module A on the selected clip.
- `POST /api/module-b`: run Module B on the selected clip.
- `POST /api/module-c`: run Module C on Module B's predicted crop when available, otherwise the selected clip.
- `POST /api/unload-model`: unload the active model and clear CUDA memory.

## Notes

The UI uses the same prompt/conversation helpers as the training and evaluation scripts, so the browser results should match the module behavior used elsewhere in the project. Module C uses Module B's predicted global start/end when available; this mirrors the intended pipeline rather than evaluating Module C on the whole manually selected clip.
