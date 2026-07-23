from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for root in (PROJECT_ROOT, SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from qwen_omd_dataloaders.datasets import EgoOopsModuleADataset, EgoOopsProvider  # noqa: E402
from qwen_omd_dataloaders.schema import WindowSpec  # noqa: E402

from run_online_pipeline_eval import (  # noqa: E402
    IOU_THRESHOLDS,
    GroundTruthEvent,
    PipelineEvent,
    compute_metrics,
    eligible_gt,
    gt_events_for_records,
    json_safe,
    module_a_config,
    val_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a clear module/end-to-end summary and confusion-matrix heatmaps from saved pipeline outputs."
    )
    parser.add_argument(
        "--pipeline-metrics-json",
        default="/home/amit/online_mistake_detection/outputs/online_pipeline_eval/full_16gb_docker/pipeline_metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <pipeline-metrics-json parent>/summary",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def event_from_row(row: dict[str, Any]) -> PipelineEvent:
    module_b = row.get("module_b_prediction")
    if isinstance(module_b, list) and len(module_b) >= 2:
        module_b_prediction = (float(module_b[0]), float(module_b[1]))
    else:
        module_b_prediction = None
    return PipelineEvent(
        video_id=str(row["video_id"]),
        task_id=str(row["task_id"]),
        step_index=int(row["step_index"]),
        instruction=str(row["instruction"]),
        trigger_time=float(row["trigger_time"]),
        accumulation_start=float(row["accumulation_start"]),
        accumulation_end=float(row["accumulation_end"]),
        module_a_prediction=str(row.get("module_a_prediction") or "COMPLETE"),
        module_b_raw=row.get("module_b_raw"),
        module_b_prediction=module_b_prediction,
        pred_global_start=None if row.get("pred_global_start") is None else float(row["pred_global_start"]),
        pred_global_end=None if row.get("pred_global_end") is None else float(row["pred_global_end"]),
        module_c_raw=row.get("module_c_raw"),
        module_c_mistake=row.get("module_c_mistake"),
        module_c_reasoning=row.get("module_c_reasoning"),
        module_a_prediction_seconds=row.get("module_a_prediction_seconds"),
        module_b_prediction_seconds=row.get("module_b_prediction_seconds"),
        module_c_prediction_seconds=row.get("module_c_prediction_seconds"),
        pipeline_prediction_seconds=row.get("pipeline_prediction_seconds"),
        matched_gt_index=row.get("matched_gt_index"),
        matched_iou=float(row.get("matched_iou") or 0.0),
        matched_gt_start=None if row.get("matched_gt_start") is None else float(row["matched_gt_start"]),
        matched_gt_end=None if row.get("matched_gt_end") is None else float(row["matched_gt_end"]),
        gt_mistake=row.get("gt_mistake"),
        gt_reasoning=row.get("gt_reasoning"),
        duplicate=bool(row.get("duplicate")),
    )


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    # Docker runs store annotation paths under /workspace/ego_oops.
    if str(path).startswith("/workspace/ego_oops/"):
        local = PROJECT_ROOT.parent / "ego_oops" / Path(*path.parts[3:])
        if local.exists():
            return local
    return path


def args_namespace(args_dict: dict[str, Any]) -> argparse.Namespace:
    namespace = argparse.Namespace(**args_dict)
    namespace.metadata = str(resolve_path(namespace.metadata))
    namespace.mistake_classes = str(resolve_path(namespace.mistake_classes))
    namespace.video_root = str(resolve_path(namespace.video_root))
    return namespace


def reconstruct_module_a_predictions(
    *,
    windows: list[WindowSpec],
    events: list[PipelineEvent],
) -> list[tuple[WindowSpec, str, str, float]]:
    trigger_keys = {
        (
            event.video_id,
            event.step_index,
            round(event.accumulation_start, 6),
            round(event.accumulation_end, 6),
        )
        for event in events
    }
    predictions: list[tuple[WindowSpec, str, str, float]] = []
    for window in windows:
        key = (
            window.video_id,
            window.step_index,
            round(window.window_start, 6),
            round(window.window_end, 6),
        )
        prediction = "COMPLETE" if key in trigger_keys else "WAIT"
        predictions.append((window, prediction, prediction, 0.0))
    return predictions


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def matrix_counts_2x2(metrics: dict[str, Any], prefix: str) -> list[list[int]] | None:
    tp = metrics.get(f"{prefix}/tp")
    fp = metrics.get(f"{prefix}/fp")
    tn = metrics.get(f"{prefix}/tn")
    fn = metrics.get(f"{prefix}/fn")
    if None in (tp, fp, tn, fn):
        return None
    # rows: GT positive / GT negative ; cols: Pred positive / Pred negative
    return [
        [int(tp), int(fn)],
        [int(fp), int(tn)],
    ]


def end_to_end_matrix(metrics: dict[str, Any], threshold: float) -> list[list[int]] | None:
    suffix = f"{threshold:.1f}"
    matrix = metrics.get(f"end_to_end_mistake_at_iou_{suffix}/confusion_matrix")
    if not isinstance(matrix, dict):
        return None
    return [
        [
            int(matrix["gt_mistake"]["pred_mistake"]),
            int(matrix["gt_mistake"]["pred_correct"]),
            int(matrix["gt_mistake"]["missed"]),
        ],
        [
            int(matrix["gt_correct"]["pred_mistake"]),
            int(matrix["gt_correct"]["pred_correct"]),
            int(matrix["gt_correct"]["missed"]),
        ],
        [
            int(matrix["unmatched_prediction"]["pred_mistake"]),
            int(matrix["unmatched_prediction"]["pred_correct"]),
            int(matrix["unmatched_prediction"]["missed"]),
        ],
    ]


def save_heatmap_png(
    *,
    matrix: list[list[int]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    values = [[float(cell) for cell in row] for row in matrix]
    fig, ax = plt.subplots(figsize=(max(5.5, 1.4 * len(col_labels) + 2), max(4.5, 1.2 * len(row_labels) + 2)))
    image = ax.imshow(values, cmap="Blues")
    ax.set_xticks(range(len(col_labels)), labels=col_labels)
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, str(value), ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_summary(
    *,
    metrics: dict[str, Any],
    events: list[PipelineEvent],
    gt_events: list[GroundTruthEvent],
) -> dict[str, Any]:
    eligible = eligible_gt(gt_events)
    summary: dict[str, Any] = {
        "overview": {
            "num_gt_steps": len(eligible),
            "num_gt_mistakes": sum(1 for gt in eligible if gt.gt_mistake),
            "num_gt_correct": sum(1 for gt in eligible if not gt.gt_mistake),
            "num_pipeline_events": len(events),
            "iou_thresholds": list(IOU_THRESHOLDS),
        },
        "module_a": {
            "what_it_measures": "WAIT vs COMPLETE on accumulating windows (reconstructed from saved triggers when offline).",
            "accuracy": metrics.get("module_a/accuracy"),
            "precision": metrics.get("module_a/precision"),
            "recall": metrics.get("module_a/recall"),
            "f1": metrics.get("module_a/f1"),
            "f_beta": metrics.get("module_a/f_beta"),
            "tp": metrics.get("module_a/tp"),
            "fp": metrics.get("module_a/fp"),
            "tn": metrics.get("module_a/tn"),
            "fn": metrics.get("module_a/fn"),
            "num_windows": metrics.get("module_a/num_windows"),
            "num_complete_predictions": metrics.get("module_a/num_complete_predictions"),
            "confusion_matrix_rows": ["GT COMPLETE", "GT WAIT"],
            "confusion_matrix_cols": ["Pred COMPLETE", "Pred WAIT"],
            "confusion_matrix": matrix_counts_2x2(metrics, "module_a"),
        },
        "module_b": {
            "what_it_measures": "Temporal localization quality for windows Module A actually triggered as COMPLETE.",
            "num_triggers": metrics.get("module_b_given_module_a_trigger/num_triggers"),
            "num_valid_windows": metrics.get("module_b_given_module_a_trigger/num_valid_windows"),
            "invalid_rate": metrics.get("module_b_given_module_a_trigger/invalid_rate"),
            "mean_iou": metrics.get("module_b_given_module_a_trigger/mean_iou") or metrics.get("mean_iou"),
            "by_iou": {
                f"{threshold:.1f}": {
                    "precision": metrics.get(f"module_b_given_module_a_trigger/precision_at_{threshold:.1f}"),
                    "recall": metrics.get(f"module_b_given_module_a_trigger/recall_at_{threshold:.1f}"),
                    "f1": metrics.get(f"module_b_given_module_a_trigger/f1_at_{threshold:.1f}"),
                }
                for threshold in IOU_THRESHOLDS
            },
        },
        "module_c_conditional": {
            "what_it_measures": (
                "Module C mistake classification only on events that already matched a GT step at IoU >= threshold. "
                "This isolates Module C from A/B misses."
            ),
            "by_iou": {
                f"{threshold:.1f}": {
                    "accuracy": metrics.get(f"module_c_given_match_iou_{threshold:.1f}/accuracy"),
                    "precision": metrics.get(f"module_c_given_match_iou_{threshold:.1f}/precision"),
                    "recall": metrics.get(f"module_c_given_match_iou_{threshold:.1f}/recall"),
                    "f1": metrics.get(f"module_c_given_match_iou_{threshold:.1f}/f1"),
                    "tp": metrics.get(f"module_c_given_match_iou_{threshold:.1f}/tp"),
                    "fp": metrics.get(f"module_c_given_match_iou_{threshold:.1f}/fp"),
                    "tn": metrics.get(f"module_c_given_match_iou_{threshold:.1f}/tn"),
                    "fn": metrics.get(f"module_c_given_match_iou_{threshold:.1f}/fn"),
                    "num_samples": metrics.get(f"module_c_given_match_iou_{threshold:.1f}/num_samples"),
                    "confusion_matrix_rows": ["GT MISTAKE", "GT CORRECT"],
                    "confusion_matrix_cols": ["Pred MISTAKE", "Pred CORRECT"],
                    "confusion_matrix": matrix_counts_2x2(metrics, f"module_c_given_match_iou_{threshold:.1f}"),
                }
                for threshold in IOU_THRESHOLDS
            },
        },
        "end_to_end_mistake_detection": {
            "what_it_measures": (
                "Full A -> B -> C mistake detection against all GT steps. "
                "A missed GT mistake (A did not trigger, or B failed, or C said correct) counts as a false negative."
            ),
            "by_iou": {
                f"{threshold:.1f}": {
                    "accuracy": metrics.get(f"end_to_end_mistake_at_iou_{threshold:.1f}/accuracy"),
                    "precision": metrics.get(f"end_to_end_mistake_at_iou_{threshold:.1f}/precision"),
                    "recall": metrics.get(f"end_to_end_mistake_at_iou_{threshold:.1f}/recall"),
                    "f1": metrics.get(f"end_to_end_mistake_at_iou_{threshold:.1f}/f1"),
                    "tp": metrics.get(f"end_to_end_mistake_at_iou_{threshold:.1f}/tp"),
                    "fp": metrics.get(f"end_to_end_mistake_at_iou_{threshold:.1f}/fp"),
                    "tn": metrics.get(f"end_to_end_mistake_at_iou_{threshold:.1f}/tn"),
                    "fn": metrics.get(f"end_to_end_mistake_at_iou_{threshold:.1f}/fn"),
                    "confusion_matrix_rows": ["GT MISTAKE", "GT CORRECT", "Unmatched prediction"],
                    "confusion_matrix_cols": ["Pred MISTAKE", "Pred CORRECT", "Missed"],
                    "confusion_matrix": end_to_end_matrix(metrics, threshold),
                }
                for threshold in IOU_THRESHOLDS
            },
        },
        "pipeline_temporal_event_detection": {
            "what_it_measures": "Whether the pipeline produced a matched event window for each GT step, ignoring mistake labels.",
            "by_iou": {
                f"{threshold:.1f}": {
                    "precision": metrics.get(f"temporal/precision_at_iou_{threshold:.1f}"),
                    "recall": metrics.get(f"temporal/recall_at_iou_{threshold:.1f}"),
                    "f1": metrics.get(f"temporal/f1_at_iou_{threshold:.1f}"),
                    "missed_step_rate": metrics.get(f"missed_step_rate_at_iou_{threshold:.1f}"),
                }
                for threshold in IOU_THRESHOLDS
            },
        },
        "prediction_runtime_seconds": {
            "module_a_mean": metrics.get("prediction_runtime/module_a/mean_seconds"),
            "module_b_mean": metrics.get("prediction_runtime/module_b/mean_seconds"),
            "module_c_mean": metrics.get("prediction_runtime/module_c/mean_seconds"),
            "pipeline_completed_action_mean": metrics.get("prediction_runtime/pipeline_completed_action/mean_seconds"),
        },
    }
    return summary


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    overview = summary["overview"]
    lines.extend(
        [
            "# Online Pipeline Evaluation Summary",
            "",
            "## How to read this file",
            "",
            "- **Module A**: WAIT vs COMPLETE on accumulating windows.",
            "- **Module B**: temporal localization for windows Module A triggered.",
            "- **Module C (conditional)**: mistake classification only when A/B already matched the GT step at IoU >= X.",
            "- **End-to-end mistake detection**: full A -> B -> C product metric over all GT steps. Missed GT mistakes count as false negatives even if Module C never saw them.",
            "- **Pipeline temporal event detection**: did A/B find/localize GT steps, ignoring mistake labels.",
            "",
            "## Overview",
            "",
            f"- GT steps: `{overview['num_gt_steps']}`",
            f"- GT mistakes: `{overview['num_gt_mistakes']}`",
            f"- GT correct: `{overview['num_gt_correct']}`",
            f"- Pipeline events: `{overview['num_pipeline_events']}`",
            f"- IoU thresholds: `{', '.join(str(x) for x in overview['iou_thresholds'])}`",
            "",
            "## Module A",
            "",
            summary["module_a"]["what_it_measures"],
            "",
            f"- Accuracy: `{pct(summary['module_a']['accuracy'])}`",
            f"- Precision: `{pct(summary['module_a']['precision'])}`",
            f"- Recall: `{pct(summary['module_a']['recall'])}`",
            f"- F1: `{pct(summary['module_a']['f1'])}`",
            f"- F-beta: `{pct(summary['module_a']['f_beta'])}`",
            f"- Confusion counts TP/FP/TN/FN: "
            f"`{summary['module_a']['tp']}/{summary['module_a']['fp']}/"
            f"{summary['module_a']['tn']}/{summary['module_a']['fn']}`",
            "",
            "Heatmap: `figures/module_a_confusion.png`",
            "",
            "## Module B",
            "",
            summary["module_b"]["what_it_measures"],
            "",
            f"- Triggers from Module A: `{summary['module_b']['num_triggers']}`",
            f"- Valid windows: `{summary['module_b']['num_valid_windows']}`",
            f"- Invalid rate: `{pct(summary['module_b']['invalid_rate'])}`",
            f"- Mean IoU: `{num(summary['module_b']['mean_iou'])}`",
            "",
            "| IoU | Precision | Recall | F1 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for threshold, values in summary["module_b"]["by_iou"].items():
        lines.append(
            f"| {threshold} | {pct(values['precision'])} | {pct(values['recall'])} | {pct(values['f1'])} |"
        )

    lines.extend(
        [
            "",
            "## Module C (conditional on matched localization)",
            "",
            summary["module_c_conditional"]["what_it_measures"],
            "",
            "| IoU | Samples | Precision | Recall | F1 | TP | FP | TN | FN | Heatmap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for threshold, values in summary["module_c_conditional"]["by_iou"].items():
        lines.append(
            f"| {threshold} | {values['num_samples']} | {pct(values['precision'])} | {pct(values['recall'])} | "
            f"{pct(values['f1'])} | {values['tp']} | {values['fp']} | {values['tn']} | {values['fn']} | "
            f"`figures/module_c_confusion_iou_{threshold}.png` |"
        )

    lines.extend(
        [
            "",
            "## End-to-end mistake detection",
            "",
            summary["end_to_end_mistake_detection"]["what_it_measures"],
            "",
            "| IoU | Precision | Recall | F1 | TP | FP | TN | FN | Heatmap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for threshold, values in summary["end_to_end_mistake_detection"]["by_iou"].items():
        lines.append(
            f"| {threshold} | {pct(values['precision'])} | {pct(values['recall'])} | {pct(values['f1'])} | "
            f"{values['tp']} | {values['fp']} | {values['tn']} | {values['fn']} | "
            f"`figures/end_to_end_confusion_iou_{threshold}.png` |"
        )

    lines.extend(
        [
            "",
            "## Pipeline temporal event detection",
            "",
            summary["pipeline_temporal_event_detection"]["what_it_measures"],
            "",
            "| IoU | Precision | Recall | F1 | Missed step rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for threshold, values in summary["pipeline_temporal_event_detection"]["by_iou"].items():
        lines.append(
            f"| {threshold} | {pct(values['precision'])} | {pct(values['recall'])} | "
            f"{pct(values['f1'])} | {pct(values['missed_step_rate'])} |"
        )

    runtime = summary["prediction_runtime_seconds"]
    lines.extend(
        [
            "",
            "## Prediction runtime (wall-clock model calls)",
            "",
            f"- Module A mean: `{num(runtime['module_a_mean'], 2)} s`",
            f"- Module B mean: `{num(runtime['module_b_mean'], 2)} s`",
            f"- Module C mean: `{num(runtime['module_c_mean'], 2)} s`",
            f"- Full completed-action mean (A+B+C): `{num(runtime['pipeline_completed_action_mean'], 2)} s`",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_heatmaps(summary: dict[str, Any], figures_dir: Path) -> list[Path]:
    written: list[Path] = []
    module_a = summary["module_a"]
    if module_a.get("confusion_matrix"):
        path = figures_dir / "module_a_confusion.png"
        save_heatmap_png(
            matrix=module_a["confusion_matrix"],
            row_labels=module_a["confusion_matrix_rows"],
            col_labels=module_a["confusion_matrix_cols"],
            title="Module A confusion matrix (COMPLETE positive)",
            output_path=path,
        )
        written.append(path)

    for threshold, values in summary["module_c_conditional"]["by_iou"].items():
        if not values.get("confusion_matrix"):
            continue
        path = figures_dir / f"module_c_confusion_iou_{threshold}.png"
        save_heatmap_png(
            matrix=values["confusion_matrix"],
            row_labels=values["confusion_matrix_rows"],
            col_labels=values["confusion_matrix_cols"],
            title=f"Module C conditional confusion (IoU >= {threshold})",
            output_path=path,
        )
        written.append(path)

    for threshold, values in summary["end_to_end_mistake_detection"]["by_iou"].items():
        if not values.get("confusion_matrix"):
            continue
        path = figures_dir / f"end_to_end_confusion_iou_{threshold}.png"
        save_heatmap_png(
            matrix=values["confusion_matrix"],
            row_labels=values["confusion_matrix_rows"],
            col_labels=values["confusion_matrix_cols"],
            title=f"End-to-end mistake confusion (IoU >= {threshold})",
            output_path=path,
        )
        written.append(path)
    return written


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.pipeline_metrics_json)
    payload = load_payload(metrics_path)
    output_dir = Path(args.output_dir) if args.output_dir else metrics_path.parent / "summary"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_args = args_namespace(payload["args"])
    split = payload["split"]
    events = [event_from_row(row) for row in payload["events"]]

    provider = EgoOopsProvider(
        metadata_path=run_args.metadata,
        mistake_classes_path=run_args.mistake_classes,
        video_root=run_args.video_root,
        video_ids=set(run_args.video_ids) if run_args.video_ids else None,
        task_ids=set(run_args.task_ids) if run_args.task_ids else None,
        max_videos=run_args.max_videos,
        require_existing_videos=True,
    )
    dataset = EgoOopsModuleADataset(provider, config=module_a_config(run_args))
    val_ids = set(split["val_video_ids"])
    windows = val_windows(dataset, val_ids, run_args.max_samples)
    gt_events = gt_events_for_records(dataset.records, val_ids)
    module_a_predictions = reconstruct_module_a_predictions(windows=windows, events=events)

    # Rematch so IoU fields stay consistent with current matching code, then recompute all metrics.
    for event in events:
        event.matched_gt_index = None
        event.matched_iou = 0.0
        event.matched_gt_start = None
        event.matched_gt_end = None
        event.gt_mistake = None
        event.gt_reasoning = None
        event.duplicate = False
    from run_online_pipeline_eval import match_events

    match_events(events, gt_events)
    metrics = compute_metrics(events, gt_events, module_a_predictions=module_a_predictions)
    summary = build_summary(metrics=metrics, events=events, gt_events=gt_events)

    summary_json = output_dir / "pipeline_summary.json"
    summary_md = output_dir / "pipeline_summary.md"
    recomputed_metrics_json = output_dir / "pipeline_metrics_recomputed.json"
    with open(summary_json, "w", encoding="utf-8") as file:
        json.dump(json_safe(summary), file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")
    with open(recomputed_metrics_json, "w", encoding="utf-8") as file:
        json.dump(json_safe(metrics), file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")
    write_markdown(summary, summary_md)
    written = write_heatmaps(summary, figures_dir)

    print(f"Saved clear summary markdown: {summary_md}")
    print(f"Saved clear summary JSON: {summary_json}")
    print(f"Saved recomputed metrics JSON: {recomputed_metrics_json}")
    print(f"Saved {len(written)} confusion-matrix heatmaps under: {figures_dir}")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
