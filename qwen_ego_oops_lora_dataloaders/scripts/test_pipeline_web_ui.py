from __future__ import annotations

import ast
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
SERVER = SCRIPT_ROOT / "pipeline_web_ui.py"
STATIC = SCRIPT_ROOT / "pipeline_web_ui_static"


def test_server_source_has_required_routes() -> None:
    source = SERVER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    route_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("/api/")
    }
    assert "/api/videos" in route_literals
    assert "/api/videos/{video_id}" in route_literals
    assert "/api/video-file/{video_id}" in route_literals
    assert "/api/module-a" in route_literals
    assert "/api/module-b" in route_literals
    assert "/api/module-c" in route_literals
    assert "/api/unload-model" in route_literals


def test_static_files_reference_expected_controls() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    for element_id in [
        "videoSelect",
        "stepSelect",
        "videoPlayer",
        "startRange",
        "endRange",
        "runAButton",
        "runBButton",
        "runCButton",
    ]:
        assert f'id="{element_id}"' in html
        assert element_id in javascript
    assert "/api/module-a" in javascript
    assert "/api/module-b" in javascript
    assert "/api/module-c" in javascript
    assert ".module-b-range" in css


def main() -> None:
    test_server_source_has_required_routes()
    test_static_files_reference_expected_controls()
    print("Pipeline web UI tests passed.")


if __name__ == "__main__":
    main()
