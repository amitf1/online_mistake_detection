from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECTION_TYPES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a flowchart-style Qwen VL LoRA architecture diagram from a PEFT checkpoint."
    )
    parser.add_argument("--module", choices=["A", "B", "C"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to a PEFT adapter checkpoint directory.")
    parser.add_argument("--model-name", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--log-to-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="qwen-omd")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-artifact-name", default=None)
    parser.add_argument("--wandb-mode", default=None)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        parser.error(f"--checkpoint does not exist: {checkpoint}")
    if not (checkpoint / "adapter_config.json").exists():
        parser.error(f"--checkpoint must contain adapter_config.json: {checkpoint}")
    return args


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def adapter_weight_files(checkpoint: Path) -> list[Path]:
    files = sorted(checkpoint.glob("adapter_model*.safetensors"))
    if files:
        return files
    files = sorted(checkpoint.glob("adapter_model*.bin"))
    if files:
        return files
    raise FileNotFoundError(f"No adapter_model safetensors/bin file found in {checkpoint}")


def tensor_shapes_from_safetensors(path: Path) -> dict[str, tuple[int, ...]]:
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "safetensors is required to read this adapter checkpoint. "
            "Run this script in the training Docker image via render_lora_architecture_docker.sh."
        ) from error

    shapes: dict[str, tuple[int, ...]] = {}
    with safe_open(path, framework="pt", device="cpu") as file:
        for key in file.keys():
            shapes[key] = tuple(int(value) for value in file.get_tensor(key).shape)
    return shapes


def tensor_shapes_from_bin(path: Path) -> dict[str, tuple[int, ...]]:
    import torch

    state = torch.load(path, map_location="cpu")
    return {
        str(key): tuple(int(value) for value in tensor.shape)
        for key, tensor in state.items()
        if hasattr(tensor, "shape")
    }


def load_adapter_tensor_shapes(checkpoint: Path) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for path in adapter_weight_files(checkpoint):
        if path.suffix == ".safetensors":
            shapes.update(tensor_shapes_from_safetensors(path))
        elif path.suffix == ".bin":
            shapes.update(tensor_shapes_from_bin(path))
    return shapes


def strip_lora_suffix(key: str) -> tuple[str, str] | None:
    marker = ".lora_A."
    if marker in key:
        return key.split(marker, 1)[0], "A"
    marker = ".lora_B."
    if marker in key:
        return key.split(marker, 1)[0], "B"
    return None


def classify_component(module_path: str) -> str:
    lowered = module_path.lower()
    if any(token in lowered for token in ("visual", "vision", "vision_tower", "image", "patch_embed")):
        return "vision_tower"
    if any(token in lowered for token in ("projector", "multi_modal", "mm_projector", "merger", "connector")):
        return "multimodal_connector"
    if any(token in lowered for token in ("self_attn", "attention", "attn")):
        return "language_attention"
    if any(token in lowered for token in ("mlp", "feed_forward")):
        return "language_mlp"
    if "lm_head" in lowered:
        return "output_head"
    if "embed" in lowered:
        return "embedding"
    return "other"


def projection_type(module_path: str) -> str:
    for projection in PROJECTION_TYPES:
        if module_path.endswith(f".{projection}") or f".{projection}." in module_path:
            return projection
    return module_path.rsplit(".", 1)[-1]


def layer_index(module_path: str) -> int | None:
    parts = module_path.split(".")
    for index, part in enumerate(parts[:-1]):
        if part in {"layers", "blocks", "h", "layer"}:
            try:
                return int(parts[index + 1])
            except ValueError:
                return None
    return None


def rank_and_params(shapes: dict[str, tuple[int, ...]]) -> tuple[int | None, int]:
    rank = None
    total = 0
    for suffix, shape in shapes.items():
        total += int(shape[0] * shape[1]) if len(shape) == 2 else 0
        if suffix == "A" and len(shape) == 2:
            rank = int(shape[0])
        elif suffix == "B" and len(shape) == 2 and rank is None:
            rank = int(shape[1])
    return rank, total


def collect_lora_modules(
    *,
    checkpoint: Path,
    adapter_config: dict[str, Any],
) -> list[dict[str, Any]]:
    tensor_shapes = load_adapter_tensor_shapes(checkpoint)
    grouped: dict[str, dict[str, tuple[int, ...]]] = defaultdict(dict)
    for key, shape in tensor_shapes.items():
        parsed = strip_lora_suffix(key)
        if parsed is None:
            continue
        module_path, suffix = parsed
        grouped[module_path][suffix] = shape

    lora_alpha = adapter_config.get("lora_alpha")
    rows = []
    for module_path, shapes in sorted(grouped.items()):
        rank, trainable_params = rank_and_params(shapes)
        component = classify_component(module_path)
        rows.append({
            "module_path": module_path,
            "component": component,
            "projection": projection_type(module_path),
            "layer_index": layer_index(module_path),
            "rank": rank,
            "lora_alpha": lora_alpha,
            "scaling": (float(lora_alpha) / rank) if rank and lora_alpha is not None else None,
            "lora_A_shape": list(shapes["A"]) if "A" in shapes else None,
            "lora_B_shape": list(shapes["B"]) if "B" in shapes else None,
            "trainable_params": trainable_params,
        })
    return rows


def summarize_lora(rows: list[dict[str, Any]], *, args: argparse.Namespace, adapter_config: dict[str, Any]) -> dict[str, Any]:
    by_component = Counter(str(row["component"]) for row in rows)
    by_projection = Counter(str(row["projection"]) for row in rows)
    by_rank = Counter(str(row["rank"]) for row in rows)
    trainable_by_component: dict[str, int] = defaultdict(int)
    layers_by_component: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        component = str(row["component"])
        trainable_by_component[component] += int(row.get("trainable_params") or 0)
        if row.get("layer_index") is not None:
            layers_by_component[component].add(int(row["layer_index"]))
    return {
        "module": args.module,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_name": args.model_name,
        "num_lora_modules": len(rows),
        "total_lora_trainable_params": sum(int(row.get("trainable_params") or 0) for row in rows),
        "components": dict(sorted(by_component.items())),
        "projections": dict(sorted(by_projection.items())),
        "ranks": dict(sorted(by_rank.items())),
        "trainable_params_by_component": dict(sorted(trainable_by_component.items())),
        "layers_by_component": {
            component: [min(layers), max(layers), len(layers)]
            for component, layers in sorted(layers_by_component.items())
            if layers
        },
        "adapter_config": {
            "peft_type": adapter_config.get("peft_type"),
            "task_type": adapter_config.get("task_type"),
            "r": adapter_config.get("r"),
            "lora_alpha": adapter_config.get("lora_alpha"),
            "target_modules": adapter_config.get("target_modules"),
            "base_model_name_or_path": adapter_config.get("base_model_name_or_path"),
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "component",
        "projection",
        "layer_index",
        "rank",
        "lora_alpha",
        "scaling",
        "trainable_params",
        "lora_A_shape",
        "lora_B_shape",
        "module_path",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def component_label(component: str) -> str:
    labels = {
        "vision_tower": "Vision Tower",
        "multimodal_connector": "Multimodal Connector",
        "language_attention": "Language Attention",
        "language_mlp": "Language MLP",
        "embedding": "Embeddings",
        "output_head": "Output Head",
        "other": "Other Modules",
    }
    return labels.get(component, component)


def component_annotation(component: str, summary: dict[str, Any]) -> str:
    count = int(summary["components"].get(component, 0))
    params = int(summary["trainable_params_by_component"].get(component, 0))
    if count == 0:
        return "No LoRA"
    layers = summary.get("layers_by_component", {}).get(component)
    layer_text = ""
    if layers:
        layer_text = f"\nlayers {layers[0]}-{layers[1]} ({layers[2]})"
    return f"LoRA modules: {count}\nparams: {params:,}{layer_text}"


def escape_svg(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_text_lines(text: str, *, x: int, y: int, line_height: int = 16, size: int = 13, weight: str = "normal") -> str:
    lines = text.splitlines() or [""]
    spans = []
    for index, line in enumerate(lines):
        spans.append(
            f'<text x="{x}" y="{y + index * line_height}" text-anchor="middle" '
            f'font-size="{size}" font-weight="{weight}" fill="#1A202C">{escape_svg(line)}</text>'
        )
    return "\n".join(spans)


def svg_box(
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    body: str,
    highlighted: bool,
) -> str:
    face = "#FFF1D6" if highlighted else "#F3F6FA"
    edge = "#D9822B" if highlighted else "#4A5568"
    stroke_width = 4 if highlighted else 2
    center = x + width // 2
    title_svg = svg_text_lines(title, x=center, y=y + 28, size=16, weight="bold")
    body_svg = svg_text_lines(body, x=center, y=y + 62, size=12)
    return f"""
<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="16" fill="{face}" stroke="{edge}" stroke-width="{stroke_width}"/>
{title_svg}
{body_svg}
""".strip()


def svg_arrow(*, x1: int, y1: int, x2: int, y2: int) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        'stroke="#2D3748" stroke-width="2.5" marker-end="url(#arrow)"/>'
    )


def render_svg(path: Path, *, summary: dict[str, Any], title: str) -> None:
    components = summary["components"]
    lora_text = (
        "LoRA adapters highlighted in orange\n"
        f"total modules: {summary['num_lora_modules']}\n"
        f"trainable params: {summary['total_lora_trainable_params']:,}\n"
        f"ranks: {', '.join(f'{key}x{value}' for key, value in summary['ranks'].items())}"
    )
    boxes = [
        svg_box(x=60, y=330, width=180, height=115, title="Video + Text", body="clip frames\ninstruction prompt", highlighted=False),
        svg_box(x=310, y=165, width=240, height=145, title="Vision Tower", body=component_annotation("vision_tower", summary), highlighted=components.get("vision_tower", 0) > 0),
        svg_box(x=630, y=165, width=250, height=145, title="Connector", body=component_annotation("multimodal_connector", summary), highlighted=components.get("multimodal_connector", 0) > 0),
        svg_box(x=970, y=330, width=220, height=115, title="Token Stream", body="visual tokens\n+ text tokens", highlighted=False),
        svg_box(x=540, y=545, width=285, height=155, title="Language Attention", body=component_annotation("language_attention", summary), highlighted=components.get("language_attention", 0) > 0),
        svg_box(x=900, y=545, width=285, height=155, title="Language MLP", body=component_annotation("language_mlp", summary), highlighted=components.get("language_mlp", 0) > 0),
        svg_box(x=1275, y=330, width=210, height=115, title="Output", body="WAIT/COMPLETE\nJSON/span/reasoning", highlighted=False),
        svg_box(x=80, y=590, width=340, height=140, title="LoRA Summary", body=lora_text, highlighted=True),
    ]
    arrows = [
        svg_arrow(x1=240, y1=388, x2=310, y2=238),
        svg_arrow(x1=550, y1=238, x2=630, y2=238),
        svg_arrow(x1=880, y1=238, x2=970, y2=388),
        svg_arrow(x1=1190, y1=388, x2=1275, y2=388),
        svg_arrow(x1=1080, y1=445, x2=682, y2=545),
        svg_arrow(x1=1080, y1=445, x2=1042, y2=545),
        svg_arrow(x1=825, y1=622, x2=900, y2=622),
        svg_arrow(x1=1185, y1=622, x2=1375, y2=445),
    ]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1540" height="820" viewBox="0 0 1540 820">
<defs>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="#2D3748"/>
  </marker>
</defs>
<rect width="1540" height="820" fill="#FFFFFF"/>
<text x="770" y="48" text-anchor="middle" font-size="26" font-weight="bold" fill="#1A202C">{escape_svg(title)}</text>
<text x="770" y="78" text-anchor="middle" font-size="14" fill="#4A5568">Checkpoint: {escape_svg(Path(summary['checkpoint']).name)} | Model: {escape_svg(summary['model_name'])}</text>
{chr(10).join(arrows)}
{chr(10).join(boxes)}
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def maybe_convert_svg_to_png(svg_path: Path, png_path: Path) -> bool:
    try:
        import cairosvg
    except ModuleNotFoundError:
        return False
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=1540, output_height=820)
    return True


def log_to_wandb(args: argparse.Namespace, output_dir: Path, files: dict[str, Path], summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or f"module_{args.module.lower()}_lora_architecture",
        mode=args.wandb_mode or None,
        config={
            "module": args.module,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "model_name": args.model_name,
        },
    )
    table = wandb.Table(columns=list(rows[0].keys()) if rows else ["module_path"])
    for row in rows:
        table.add_data(*(row.get(column) for column in table.columns))
    log_payload: dict[str, Any] = {
        "lora_architecture/modules": table,
        "lora_architecture/num_lora_modules": summary["num_lora_modules"],
        "lora_architecture/total_lora_trainable_params": summary["total_lora_trainable_params"],
    }
    if files.get("png") and files["png"].exists():
        log_payload["lora_architecture/diagram"] = wandb.Image(str(files["png"]))
    run.log(log_payload)
    artifact_name = args.wandb_artifact_name or f"module-{args.module.lower()}-lora-architecture"
    artifact = wandb.Artifact(artifact_name, type="lora-architecture", metadata=summary)
    artifact.add_dir(str(output_dir))
    run.log_artifact(artifact)
    run.finish()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_config = load_json(checkpoint / "adapter_config.json")
    rows = collect_lora_modules(checkpoint=checkpoint, adapter_config=adapter_config)
    if not rows:
        raise SystemExit(f"No LoRA adapter tensors found in checkpoint: {checkpoint}")
    summary = summarize_lora(rows, args=args, adapter_config=adapter_config)
    title = args.title or f"Module {args.module} Qwen VL LoRA Architecture"

    files = {
        "svg": output_dir / "lora_architecture.svg",
        "png": output_dir / "lora_architecture.png",
        "csv": output_dir / "lora_modules.csv",
        "json": output_dir / "lora_modules.json",
        "summary": output_dir / "lora_summary.json",
    }
    render_svg(files["svg"], summary=summary, title=title)
    png_written = maybe_convert_svg_to_png(files["svg"], files["png"])
    if not png_written:
        print("PNG conversion skipped because cairosvg is not installed; SVG output was written.")
    write_csv(files["csv"], rows)
    write_json(files["json"], rows)
    write_json(files["summary"], summary)

    print("LoRA architecture summary:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Wrote outputs:")
    for path in files.values():
        if path.exists():
            print(f"- {path}")

    if args.log_to_wandb:
        log_to_wandb(args, output_dir, files, summary, rows)


if __name__ == "__main__":
    main()
