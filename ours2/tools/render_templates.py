#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

OUTPUTS = (
    ("aie/ProcessGraph/StencilCoreGraph.h.j2", "aie/ProcessGraph/StencilCoreGraph.h"),
    ("pl/TopPL.cpp.j2", "pl/TopPL.cpp"),
    ("pl/Q0Ctrl.cpp.j2", "pl/Q0Ctrl.cpp"),
    ("conn.cfg.j2", "conn.cfg"),
)


def read_design(path: Path) -> dict[str, object]:
    design: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        design[key.strip()] = int(value) if value.isdigit() else value

    required = {
        "graph_id",
        "rows",
        "cols",
        "toppl_workers",
        "worker_lanes",
        "parallel_lanes",
        "aie_rows",
        "aie_cols",
        "kernels_per_lane",
    }
    missing = sorted(required - set(design))
    if missing:
        raise RuntimeError(f"{path} missing required keys: {', '.join(missing)}")

    design["workers"] = list(range(int(design["toppl_workers"])))
    design["worker_lanes_list"] = list(range(int(design["worker_lanes"])))
    design["lanes"] = list(range(int(design["parallel_lanes"])))
    design["worker_lane_pairs"] = [
        {
            "worker": worker,
            "local_lane": local_lane,
            "global_lane": worker * int(design["worker_lanes"]) + local_lane,
        }
        for worker in design["workers"]
        for local_lane in design["worker_lanes_list"]
    ]
    return design


def main() -> int:
    parser = argparse.ArgumentParser(description="Render ours2 Jinja templates.")
    parser.add_argument("--check", action="store_true", help="fail if outputs differ")
    args = parser.parse_args()

    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError as exc:
        raise SystemExit("jinja2 is required: python -m pip install jinja2") from exc

    design = read_design(ROOT / "design.yaml")
    env = Environment(
        loader=FileSystemLoader(ROOT / "templates"),
        keep_trailing_newline=True,
        lstrip_blocks=True,
        trim_blocks=True,
    )

    changed = []
    for template_name, output_name in OUTPUTS:
        rendered = env.get_template(template_name).render(**design)
        output_path = ROOT / output_name
        if args.check:
            current = output_path.read_text(encoding="utf-8")
            if current != rendered:
                changed.append(output_name)
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")

    if changed:
        for name in changed:
            print(f"out of date: {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
