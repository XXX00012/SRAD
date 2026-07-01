# Codex Guidance For SRAD ours_192lane

This repository is a local source workspace for a Versal/VCK190 SRAD AI Engine project. Treat it as static-analysis-first unless the user explicitly asks for a specific local command.

## Persistent Rules

- Answer Chinese user requests in Chinese.
- Do not run Vitis, AIE, HLS, XRT, simulator, hardware emulation, board, package, or project build commands locally unless the user explicitly requests that exact command.
- Before changing AIE, PL, host, connectivity, Makefile, or data-generation code, first explain the exact files/functions to change and wait for explicit confirmation.
- Use targeted searches in local manuals under `D:/科研/文章/references_md`; do not load whole manuals into context.
- Preserve user edits and uncommitted work. Do not revert unrelated changes.

## Preferred Workflow

For any non-trivial error or change request, use `.agents/skills/aie-debug-workflow`.

Main agent responsibilities:

- Define the problem and affected domains.
- Dispatch focused subagents for independent domains.
- Merge findings and choose the smallest patch plan.
- Ask for confirmation before edits.
- After edits, provide remote validation commands for the user to run.

Subagent domains:

- Manual lookup: AMD manuals and relevant papers only.
- AIE graph/kernel: `aie/**`.
- PL/HLS: `pl/**`.
- Host/runtime: `ps/host.cpp`.
- Connectivity/build: `conn.cfg`, `pl/*.cfg`, `Makefile`, plus name checks against `aie/TopGraph.*` and `ps/host.cpp`.
- Data verification: `data/**`.

## Project Invariants

- Row-stream SRAD design, not old tile+halo design.
- Input row to AIE is 258 floats: 256 data + q0 + pad.
- Output row from AIE is 256 floats.
- PLIO data width stays 64-bit unless the user confirms a redesign.
- Q0Ctrl feedback loop stays in the design.
- q0 should not be changed to s_axilite or RTP without explicit confirmation.
- Multi-iteration ping-pong behavior should be preserved.
- `data/gen_case.py` and host CPU reference logic are the numerical semantics reference.

## Manual Search Routing

- `UG1079.md`: ADF graph, kernels, buffers, margins, PLIO/GMIO, pktstream, profiling.
- `UG1399.md`: Vitis compile/link/package, system config, XRT host flow.
- `AM009.md`: Versal architecture and AIE array context.
- `UG1076.md`: platform/runtime/board environment.

Suggested pattern:

```powershell
rg -n "<error text|API|concept>" "D:/科研/文章/references_md/UG1079.md" "D:/科研/文章/references_md/UG1399.md" "D:/科研/文章/references_md/UG1076.md" "D:/科研/文章/references_md/AM009.md"
```

## Response Shape For AIE Issues

Use this structure when the task is non-trivial:

- `现状判断`: what local code and logs indicate.
- `分工结果`: concise summary from subagents, if used.
- `建议改动`: exact files/functions and logic changes.
- `依据`: local code references and manual hits.
- `风险/验证`: what the user should validate remotely.
- `需要确认`: ask before editing when changes are proposed.
