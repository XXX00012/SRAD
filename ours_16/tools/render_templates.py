#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

OUTPUTS = (
    ("aie/Config.h.j2", "aie/Config.h"),
    ("aie/ProcessGraph/StencilCoreGraph.h.j2", "aie/ProcessGraph/StencilCoreGraph.h"),
    ("pl/TopPL.cpp.j2", "pl/TopPL.cpp"),
    ("pl/Q0Ctrl.cpp.j2", "pl/Q0Ctrl.cpp"),
    ("conn.cfg.j2", "conn.cfg"),
)


def to_attr(value: object) -> object:
    if isinstance(value, dict):
        return SimpleNamespace(
            **{str(key): to_attr(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [to_attr(item) for item in value]
    return value


def eval_expr(expr: str, ctx: dict[str, object]) -> object:
    safe_globals = {
        "__builtins__": {},
        "int": int,
        "len": len,
        "range": range,
        "str": str,
    }
    return eval(expr, safe_globals, ctx)


def tokenize_template(text: str) -> list[str]:
    tokens = re.split(r"(\{\{.*?\}\}|\{%.*?%\})", text, flags=re.S)
    for idx, token in enumerate(tokens):
        if not token.startswith("{%"):
            continue

        if idx > 0 and not tokens[idx - 1].startswith(("{{", "{%")):
            prev = tokens[idx - 1]
            line_start = max(prev.rfind("\n"), prev.rfind("\r")) + 1
            if prev[line_start:].strip(" \t") == "":
                tokens[idx - 1] = prev[:line_start]

        if idx + 1 < len(tokens) and not tokens[idx + 1].startswith(
            ("{{", "{%")
        ):
            if tokens[idx + 1].startswith("\r\n"):
                tokens[idx + 1] = tokens[idx + 1][2:]
            elif tokens[idx + 1].startswith("\n"):
                tokens[idx + 1] = tokens[idx + 1][1:]

    return tokens


def find_matching(tokens: list[str], start: int, end_tag: str) -> int:
    depth = 0
    for idx in range(start, len(tokens)):
        token = tokens[idx]
        if not token.startswith("{%"):
            continue
        tag = token[2:-2].strip()
        if tag.startswith("for ") or tag.startswith("if "):
            depth += 1
        elif tag in ("endfor", "endif"):
            if depth == 0 and tag == end_tag:
                return idx
            depth -= 1
    raise RuntimeError(f"missing {end_tag} in template")


def render_tokens(tokens: list[str],
                  start: int,
                  stop: int,
                  ctx: dict[str, object]) -> str:
    output: list[str] = []
    idx = start
    while idx < stop:
        token = tokens[idx]
        if token.startswith("{{"):
            output.append(str(eval_expr(token[2:-2].strip(), ctx)))
            idx += 1
            continue
        if not token.startswith("{%"):
            output.append(token)
            idx += 1
            continue

        tag = token[2:-2].strip()
        if tag.startswith("for "):
            match = re.fullmatch(r"for\s+([A-Za-z_]\w*)\s+in\s+(.+)", tag)
            if not match:
                raise RuntimeError(f"unsupported for tag: {tag}")
            var_name = match.group(1)
            iterable = list(eval_expr(match.group(2), ctx))
            end_idx = find_matching(tokens, idx + 1, "endfor")
            old_var = ctx.get(var_name)
            had_var = var_name in ctx
            old_loop = ctx.get("loop")
            had_loop = "loop" in ctx
            for item_idx, item in enumerate(iterable):
                ctx[var_name] = to_attr(item)
                ctx["loop"] = SimpleNamespace(
                    index=item_idx + 1,
                    index0=item_idx,
                    first=item_idx == 0,
                    last=item_idx == len(iterable) - 1,
                    length=len(iterable),
                )
                output.append(render_tokens(tokens, idx + 1, end_idx, ctx))
            if had_var:
                ctx[var_name] = old_var
            else:
                ctx.pop(var_name, None)
            if had_loop:
                ctx["loop"] = old_loop
            else:
                ctx.pop("loop", None)
            idx = end_idx + 1
            continue

        if tag.startswith("if "):
            end_idx = find_matching(tokens, idx + 1, "endif")
            if eval_expr(tag[3:].strip(), ctx):
                output.append(render_tokens(tokens, idx + 1, end_idx, ctx))
            idx = end_idx + 1
            continue

        if tag in ("endif", "endfor"):
            raise RuntimeError(f"unexpected {tag}")
        raise RuntimeError(f"unsupported tag: {tag}")

    return "".join(output)


class FallbackTemplate:
    def __init__(self, path: Path):
        self.path = path

    def render(self, **ctx: object) -> str:
        text = self.path.read_text(encoding="utf-8")
        render_ctx = {key: to_attr(value) for key, value in ctx.items()}
        tokens = tokenize_template(text)
        return render_tokens(tokens, 0, len(tokens), render_ctx)


class FallbackEnvironment:
    def __init__(self, template_root: Path):
        self.template_root = template_root

    def get_template(self, template_name: str) -> FallbackTemplate:
        return FallbackTemplate(self.template_root / template_name)


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
        "worker_group_lanes",
        "parallel_lanes",
        "aie_rows",
        "aie_cols",
        "kernels_per_lane",
    }
    missing = sorted(required - set(design))
    if missing:
        raise RuntimeError(f"{path} missing required keys: {', '.join(missing)}")

    toppl_workers = int(design["toppl_workers"])
    worker_lanes = int(design["worker_lanes"])
    worker_group_lanes = int(design["worker_group_lanes"])
    parallel_lanes = int(design["parallel_lanes"])
    output_lanes_per_plio = 2
    if toppl_workers * worker_lanes != parallel_lanes:
        raise RuntimeError(
            "parallel_lanes must equal toppl_workers * worker_lanes "
            f"({parallel_lanes} != {toppl_workers} * {worker_lanes})"
        )
    if worker_lanes % worker_group_lanes != 0:
        raise RuntimeError(
            "worker_lanes must be divisible by worker_group_lanes "
            f"({worker_lanes} % {worker_group_lanes} != 0)"
        )
    if parallel_lanes % output_lanes_per_plio != 0:
        raise RuntimeError(
            "parallel_lanes must be divisible by output_lanes_per_plio "
            f"({parallel_lanes} % {output_lanes_per_plio} != 0)"
        )
    if worker_lanes % output_lanes_per_plio != 0:
        raise RuntimeError(
            "worker_lanes must be divisible by output_lanes_per_plio "
            f"({worker_lanes} % {output_lanes_per_plio} != 0)"
        )
    if worker_group_lanes % output_lanes_per_plio != 0:
        raise RuntimeError(
            "worker_group_lanes must be divisible by output_lanes_per_plio "
            f"({worker_group_lanes} % {output_lanes_per_plio} != 0)"
        )

    design["workers"] = list(range(int(design["toppl_workers"])))
    design["worker_lanes_list"] = list(range(int(design["worker_lanes"])))
    design["worker_group_lanes_list"] = list(range(worker_group_lanes))
    design["worker_groups"] = list(range(worker_lanes // worker_group_lanes))
    design["lanes"] = list(range(int(design["parallel_lanes"])))
    design["output_lanes_per_plio"] = output_lanes_per_plio
    design["output_plio_groups"] = parallel_lanes // output_lanes_per_plio
    design["output_plio_groups_list"] = list(
        range(int(design["output_plio_groups"]))
    )
    design["worker_output_groups"] = worker_lanes // output_lanes_per_plio
    design["worker_output_groups_list"] = list(
        range(int(design["worker_output_groups"]))
    )
    design["worker_lane_pairs"] = [
        {
            "worker": worker,
            "local_lane": local_lane,
            "global_lane": worker * int(design["worker_lanes"]) + local_lane,
        }
        for worker in design["workers"]
        for local_lane in design["worker_lanes_list"]
    ]
    design["worker_output_pairs"] = [
        {
            "worker": worker,
            "local_output_group": output_group,
            "global_output_group":
                worker * int(design["worker_output_groups"]) + output_group,
        }
        for worker in design["workers"]
        for output_group in design["worker_output_groups_list"]
    ]
    return design


def main() -> int:
    parser = argparse.ArgumentParser(description="Render ours_16 Jinja templates.")
    parser.add_argument("--check", action="store_true", help="fail if outputs differ")
    args = parser.parse_args()

    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError:
        Environment = None
        FileSystemLoader = None

    design = read_design(ROOT / "design.yaml")
    if Environment is not None:
        env = Environment(
            loader=FileSystemLoader(ROOT / "templates"),
            keep_trailing_newline=True,
            lstrip_blocks=True,
            trim_blocks=True,
        )
    else:
        env = FallbackEnvironment(ROOT / "templates")

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
