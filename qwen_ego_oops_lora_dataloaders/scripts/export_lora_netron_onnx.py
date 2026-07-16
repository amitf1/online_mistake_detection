from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from render_lora_architecture import (
    collect_lora_modules,
    load_json,
    summarize_lora,
    write_csv,
    write_json,
)


CUSTOM_DOMAIN = "com.online_mistake_detection.lora"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a lightweight, Netron-friendly ONNX graph showing where LoRA adapters "
            "are attached in a saved PEFT checkpoint. This is for architecture inspection, "
            "not for inference."
        )
    )
    parser.add_argument("--module", choices=["A", "B", "C"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to PEFT adapter checkpoint directory.")
    parser.add_argument("--model-name", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--view",
        choices=("summary", "detailed"),
        default="summary",
        help="summary groups LoRA adapters into major blocks; detailed draws every adapter module.",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        parser.error(f"--checkpoint does not exist: {checkpoint}")
    if not (checkpoint / "adapter_config.json").exists():
        parser.error(f"--checkpoint must contain adapter_config.json: {checkpoint}")
    return args


def require_onnx() -> Any:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "The 'onnx' Python package is required for Netron export. "
            "Install it with `python -m pip install onnx`, or run the Docker wrapper with "
            "ONNX_AUTO_INSTALL=true."
        ) from error
    return onnx, helper, TensorProto


def clean_name(value: Any) -> str:
    text = str(value if value is not None else "unknown")
    safe = []
    for char in text:
        if char.isalnum() or char in "._-":
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "unknown"


def shape_to_text(shape: Any) -> str:
    if not shape:
        return ""
    return "x".join(str(item) for item in shape)


def row_attributes(row: dict[str, Any]) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "module_path": str(row["module_path"]),
        "component": str(row["component"]),
        "projection": str(row["projection"]),
        "trainable_params": int(row.get("trainable_params") or 0),
    }
    if row.get("layer_index") is not None:
        attrs["layer_index"] = int(row["layer_index"])
    if row.get("rank") is not None:
        attrs["rank"] = int(row["rank"])
    if row.get("lora_alpha") is not None:
        attrs["lora_alpha"] = int(row["lora_alpha"])
    if row.get("scaling") is not None:
        attrs["scaling"] = float(row["scaling"])
    if row.get("lora_A_shape") is not None:
        attrs["lora_A_shape"] = shape_to_text(row["lora_A_shape"])
    if row.get("lora_B_shape") is not None:
        attrs["lora_B_shape"] = shape_to_text(row["lora_B_shape"])
    return attrs


def build_onnx_model(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> Any:
    onnx, helper, TensorProto = require_onnx()
    nodes = []
    inputs = []
    outputs = []
    seen_inputs: set[str] = set()

    for index, row in enumerate(rows):
        component = clean_name(row["component"])
        projection = clean_name(row["projection"])
        layer = clean_name(row.get("layer_index"))
        base_name = f"{index:03d}_{component}_layer{layer}_{projection}"
        component_input = f"{component}_hidden_state"
        if component_input not in seen_inputs:
            inputs.append(helper.make_tensor_value_info(component_input, TensorProto.FLOAT, ["batch", "hidden"]))
            seen_inputs.add(component_input)

        base_out = f"{base_name}_base_linear_out"
        lora_a_out = f"{base_name}_lora_A_out"
        lora_b_out = f"{base_name}_lora_B_out"
        scaled_out = f"{base_name}_scaled_lora_out"
        adapted_out = f"{base_name}_adapted_out"
        attrs = row_attributes(row)

        nodes.append(
            helper.make_node(
                "BaseLinear",
                inputs=[component_input],
                outputs=[base_out],
                domain=CUSTOM_DOMAIN,
                **attrs,
            )
        )
        nodes.append(
            helper.make_node(
                "LoRA_A",
                inputs=[component_input],
                outputs=[lora_a_out],
                domain=CUSTOM_DOMAIN,
                **attrs,
            )
        )
        nodes.append(
            helper.make_node(
                "LoRA_B",
                inputs=[lora_a_out],
                outputs=[lora_b_out],
                domain=CUSTOM_DOMAIN,
                **attrs,
            )
        )
        nodes.append(
            helper.make_node(
                "Scale",
                inputs=[lora_b_out],
                outputs=[scaled_out],
                domain=CUSTOM_DOMAIN,
                scaling=float(row.get("scaling") or 1.0),
            )
        )
        nodes.append(
            helper.make_node(
                "AddLoRA",
                inputs=[base_out, scaled_out],
                outputs=[adapted_out],
                domain=CUSTOM_DOMAIN,
                **attrs,
            )
        )
        outputs.append(helper.make_tensor_value_info(adapted_out, TensorProto.FLOAT, ["batch", "hidden"]))

    graph = helper.make_graph(
        nodes=nodes,
        name=clean_name(args.title or f"Module {args.module} LoRA Netron graph"),
        inputs=inputs,
        outputs=outputs,
    )
    model = helper.make_model(
        graph,
        producer_name="online_mistake_detection_lora_netron_export",
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid(CUSTOM_DOMAIN, 1),
        ],
    )
    helper.set_model_props(
        model,
        {
            "purpose": "Netron architecture visualization only; not an executable inference graph.",
            "module": args.module,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "model_name": args.model_name,
            "num_lora_modules": str(summary["num_lora_modules"]),
            "total_lora_trainable_params": str(summary["total_lora_trainable_params"]),
            "components": json.dumps(summary["components"], sort_keys=True),
            "projections": json.dumps(summary["projections"], sort_keys=True),
        },
    )
    return onnx, model


def component_rows(rows: list[dict[str, Any]], component: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["component"] == component]


def projection_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        projection = str(row["projection"])
        counts[projection] = counts.get(projection, 0) + 1
    return dict(sorted(counts.items()))


def layer_summary(rows: list[dict[str, Any]]) -> str:
    layers = sorted({int(row["layer_index"]) for row in rows if row.get("layer_index") is not None})
    if not layers:
        return ""
    return f"{layers[0]}-{layers[-1]} ({len(layers)} layers)"


def total_params(rows: list[dict[str, Any]]) -> int:
    return sum(int(row.get("trainable_params") or 0) for row in rows)


def rank_summary(rows: list[dict[str, Any]]) -> str:
    ranks = sorted({str(row.get("rank")) for row in rows if row.get("rank") is not None})
    return ",".join(ranks)


def summary_node_attrs(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    return {
        "label": label,
        "lora_modules": len(rows),
        "lora_trainable_params": total_params(rows),
        "ranks": rank_summary(rows),
        "layers": layer_summary(rows),
        "projections": json.dumps(projection_counts(rows), sort_keys=True),
    }


def build_summary_onnx_model(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> Any:
    onnx, helper, TensorProto = require_onnx()
    vision_rows = component_rows(rows, "vision_tower")
    attention_rows = component_rows(rows, "language_attention")
    mlp_rows = component_rows(rows, "language_mlp")
    other_rows = [
        row
        for row in rows
        if row["component"] not in {"vision_tower", "language_attention", "language_mlp"}
    ]

    task_output = "Temporal JSON" if args.module == "B" else "WAIT / COMPLETE"
    input_name = "video_clip_and_prompt"
    vision_out = "vision_features"
    connector_out = "multimodal_tokens"
    attention_out = "language_attention_state"
    mlp_out = "language_mlp_state"
    output_name = "model_output"

    nodes = [
        helper.make_node(
            "VisionTowerWithLoRA",
            inputs=[input_name],
            outputs=[vision_out],
            domain=CUSTOM_DOMAIN,
            **summary_node_attrs(vision_rows, "Vision Tower + LoRA adapters"),
        ),
        helper.make_node(
            "MultimodalConnector",
            inputs=[vision_out],
            outputs=[connector_out],
            domain=CUSTOM_DOMAIN,
            label="Vision-to-language token projection",
            lora_modules=len(component_rows(rows, "multimodal_connector")),
            lora_trainable_params=total_params(component_rows(rows, "multimodal_connector")),
        ),
        helper.make_node(
            "LanguageAttentionWithLoRA",
            inputs=[connector_out],
            outputs=[attention_out],
            domain=CUSTOM_DOMAIN,
            **summary_node_attrs(attention_rows, "Language attention + LoRA adapters"),
        ),
        helper.make_node(
            "LanguageMLPWithLoRA",
            inputs=[attention_out],
            outputs=[mlp_out],
            domain=CUSTOM_DOMAIN,
            **summary_node_attrs(mlp_rows, "Language MLP + LoRA adapters"),
        ),
        helper.make_node(
            "TaskOutput",
            inputs=[mlp_out],
            outputs=[output_name],
            domain=CUSTOM_DOMAIN,
            label=task_output,
            lora_modules=len(other_rows),
            lora_trainable_params=total_params(other_rows),
        ),
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name=clean_name(args.title or f"Module {args.module} Simplified LoRA Netron graph"),
        inputs=[helper.make_tensor_value_info(input_name, TensorProto.FLOAT, ["video", "text"])],
        outputs=[helper.make_tensor_value_info(output_name, TensorProto.FLOAT, ["tokens"])],
    )
    model = helper.make_model(
        graph,
        producer_name="online_mistake_detection_lora_netron_export",
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid(CUSTOM_DOMAIN, 1),
        ],
    )
    helper.set_model_props(
        model,
        {
            "purpose": "Simplified Netron architecture visualization only; not an executable inference graph.",
            "module": args.module,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "model_name": args.model_name,
            "num_lora_modules": str(summary["num_lora_modules"]),
            "total_lora_trainable_params": str(summary["total_lora_trainable_params"]),
            "components": json.dumps(summary["components"], sort_keys=True),
        },
    )
    return onnx, model


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_config = load_json(checkpoint / "adapter_config.json")
    rows = collect_lora_modules(checkpoint=checkpoint, adapter_config=adapter_config)
    summary = summarize_lora(rows, args=args, adapter_config=adapter_config)
    if args.view == "summary":
        onnx, model = build_summary_onnx_model(args=args, rows=rows, summary=summary)
        onnx_name = "lora_netron_simplified.onnx"
    else:
        onnx, model = build_onnx_model(args=args, rows=rows, summary=summary)
        onnx_name = "lora_netron_detailed.onnx"

    onnx_path = output_dir / onnx_name
    onnx.save(model, onnx_path)
    write_csv(output_dir / "lora_modules.csv", rows)
    write_json(output_dir / "lora_modules.json", rows)
    write_json(output_dir / "lora_summary.json", summary)

    manifest = {
        "onnx": str(onnx_path),
        "note": (
            f"Open {onnx_name} in Netron. This graph shows LoRA placement and shapes, "
            "not full Qwen inference."
        ),
        "view": args.view,
        "summary": summary,
    }
    write_json(output_dir / "netron_export_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
