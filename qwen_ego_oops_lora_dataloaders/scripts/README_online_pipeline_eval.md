# Online Pipeline Evaluation

`run_online_pipeline_eval.py` evaluates the full online product path:

```text
accumulating window -> Module A COMPLETE -> Module B temporal crop -> Module C mistake JSON
```

It reports the final pipeline output, not only the isolated module scores.

## Required Checkpoints

Pass local PEFT checkpoint directories for all three modules:

- `--module-a-checkpoint`
- `--module-b-checkpoint`
- `--module-c-checkpoint`

The script loads one model phase at a time: Module A over validation windows, then Module B over triggered windows, then Module C over valid Module B crops. This is slower than keeping all models resident, but it is practical on 16GB GPUs.

## Local 16GB Smoke Command

Use this first to verify paths and outputs:

```bash
MAX_SAMPLES=25 \
MAX_VIDEOS=10 \
VAL_VIDEOS_PER_TASK=1 \
OUTPUT_DIR=/home/amit/online_mistake_detection/outputs/online_pipeline_eval/smoke_16gb_docker \
bash scripts/run_online_pipeline_eval_docker.sh
```

The wrapper defaults to the locally downloaded best checkpoints:

```text
MODULE_A_CHECKPOINT=/home/amit/online_mistake_detection/outputs/module_a_qwen35_2b_lora_wait_complete_vision/runs/module_a_recall_loss2_from_ep8_to_ep50_v8/best_ep10
MODULE_B_CHECKPOINT=/home/amit/online_mistake_detection/outputs/wandb_artifacts/module-b-9kahxivi-best_v1/best_ep7
MODULE_C_CHECKPOINT=/home/amit/online_mistake_detection/outputs/module_c_qwen35_lora_reasoning/runs/module_c_16gb_2b_r16_16f384_seq3072/best_ep1
```

Override those environment variables if you want different checkpoints.

Equivalent non-Docker command:

```bash
python3 scripts/run_online_pipeline_eval.py \
  --module-a-checkpoint /path/to/module_a_checkpoint \
  --module-b-checkpoint /path/to/module_b_checkpoint \
  --module-c-checkpoint /path/to/module_c_checkpoint \
  --video-root /home/amit/online_mistake_detection/data/videos-processed-720p \
  --max-videos 10 \
  --val-videos-per-task 1 \
  --max-samples 25 \
  --max-frames-a 16 \
  --max-frames-b 16 \
  --max-frames-c 16 \
  --vision-resize-a 336 \
  --vision-resize-b 384 \
  --vision-resize-c 384 \
  --max-seq-length-a 3072 \
  --max-seq-length-b 5120 \
  --max-seq-length-c 3072 \
  --output-dir /home/amit/online_mistake_detection/outputs/online_pipeline_eval/smoke_16gb
```

For a fuller 16GB run, remove `--max-samples` after the smoke test. If memory is tight, lower `--max-frames-b` first because Module B usually has the longest clip/prompt pressure.

Docker full 16GB validation command:

```bash
OUTPUT_DIR=/home/amit/online_mistake_detection/outputs/online_pipeline_eval/full_16gb_docker \
bash scripts/run_online_pipeline_eval_docker.sh
```

Module C defaults match training (`16` frames @ `384`). To re-run only Module C on saved A/B windows after a settings fix:

```bash
OUTPUT_DIR=/home/amit/online_mistake_detection/outputs/online_pipeline_eval/full_16gb_docker_c16f384 \
REUSE_EVENTS_JSON=/home/amit/online_mistake_detection/outputs/online_pipeline_eval/full_16gb_docker/pipeline_metrics.json \
bash scripts/run_online_pipeline_eval_docker.sh
```

## 48GB / Vast.ai Command

Use larger frame budgets and sequence length for Module B, matching the successful Module B 48GB direction:

```bash
python3 scripts/run_online_pipeline_eval.py \
  --module-a-checkpoint /path/to/module_a_checkpoint \
  --module-b-checkpoint /path/to/module_b_checkpoint \
  --module-c-checkpoint /path/to/module_c_checkpoint \
  --video-root /home/amit/online_mistake_detection/data/videos-processed-720p \
  --max-videos 50 \
  --val-videos-per-task 2 \
  --max-frames-a 32 \
  --max-frames-b 24 \
  --max-frames-c 16 \
  --vision-resize-a 384 \
  --vision-resize-b 512 \
  --vision-resize-c 384 \
  --max-seq-length-a 6144 \
  --max-seq-length-b 8192 \
  --max-seq-length-c 4096 \
  --output-dir /workspace/online_mistake_detection/outputs/online_pipeline_eval/full_48gb
```

## Outputs

The output directory contains:

- `pipeline_metrics.json`: args, split details, aggregate metrics, and event list.
- `pipeline_events.jsonl`: one final pipeline event per detected attempt.
- `pipeline_events.csv`: flat audit table for spreadsheet review.

Important per-event fields include `trigger_time`, `pred_global_start`, `pred_global_end`, `module_c_mistake`, `module_c_reasoning`, `matched_gt_start`, `matched_gt_end`, `matched_iou`, `gt_mistake`, and `duplicate`.

Prediction runtime fields are also saved per event:

- `module_a_prediction_seconds`
- `module_b_prediction_seconds`
- `module_c_prediction_seconds`
- `pipeline_prediction_seconds`

`pipeline_prediction_seconds` is the sum of A + B + C prediction calls for a completed pipeline event. It excludes one-time checkpoint loading overhead because the batch evaluator loads each module once and then runs all examples for that module.

## Metric Definitions

Temporal `recall_at_iou_X` asks: for each eligible ground-truth procedural segment, did the full pipeline produce a matched final window for the same video and step with IoU at least `X`?

Latency is measured as:

```text
trigger_time - matched_gt_end
```

Negative latency means Module A triggered before the annotated step end. Positive latency means the online pipeline detected after completion. The output includes `latency/mean_seconds`, `latency/median_seconds`, `latency/p90_seconds`, `latency/p95_seconds`, `latency/min_seconds`, `latency/max_seconds`, `latency/mean_absolute_seconds`, and early/on-time/late counts and rates. The same family is also reported per temporal threshold as `latency_at_iou_0.1/...`, `latency_at_iou_0.3/...`, and `latency_at_iou_0.5/...`.

Prediction runtime is different from temporal latency. It measures how long the model calls took on the machine. The output includes:

- `prediction_runtime/module_a/...`
- `prediction_runtime/module_b/...`
- `prediction_runtime/module_c/...`
- `prediction_runtime/pipeline_completed_action/...`

Each family includes count, mean, median, p90, p95, min, and max seconds.

`end_to_end_mistake_recall_at_iou_0.5` asks: among all ground-truth mistake segments, how many were detected by the entire chain with a matched window at IoU >= 0.5 and Module C predicted `mistake=true`?

`end_to_end_mistake_precision_at_iou_0.5` asks: among all pipeline events where Module C predicted `mistake=true`, how many matched a ground-truth mistake segment at IoU >= 0.5?

`end_to_end_mistake_f1_at_iou_0.5` is the harmonic mean of that end-to-end mistake precision and recall.

`mistake_f1_given_match_iou_0.5` isolates Module C: it only evaluates events that already matched a ground-truth segment with IoU >= 0.5. This answers whether Module C classified mistake/correct correctly once A/B found the right time window.

The script saves two confusion-matrix families at each IoU threshold:

- `module_c_given_match_iou_X/confusion_matrix`: only matched events at that IoU.
- `end_to_end_mistake_at_iou_X/confusion_matrix`: all ground-truth segments plus unmatched predictions, making missed GT mistakes and unmatched predicted mistakes visible.

## Notes

The first implementation evaluates eligible procedural segments where `instruction_index >= 0`. Undefined extra mistake actions are counted as ineligible because the current A -> B -> C pipeline is step-instruction driven and has no step prompt for those segments.

Use `--skip-inference` to verify split/ground-truth loading without loading checkpoints.

## Clear Summary And Confusion Heatmaps

After a pipeline run, rebuild a readable summary plus heatmaps without reloading models:

```bash
bash scripts/summarize_online_pipeline_eval_docker.sh
```

This writes:

```text
outputs/online_pipeline_eval/full_16gb_docker/summary/pipeline_summary.md
outputs/online_pipeline_eval/full_16gb_docker/summary/pipeline_summary.json
outputs/online_pipeline_eval/full_16gb_docker/summary/pipeline_metrics_recomputed.json
outputs/online_pipeline_eval/full_16gb_docker/summary/figures/*.png
```

The recomputed metrics include IoU thresholds `0.1`, `0.3`, `0.5`, `0.7`, and `0.9`. The markdown summary separates Module A, Module B, conditional Module C, end-to-end mistake detection, and pipeline temporal event detection.
