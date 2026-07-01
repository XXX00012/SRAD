---
name: srad-aie-debug-workflow
description: Use when analyzing SRAD ours_192lane AIE, Vitis, PLIO, HLS, XRT, conn.cfg, Makefile, host, data generation, compiler, simulator, or board-runtime issues in this project.
---

# SRAD AIE Debug Workflow

## Core Rule

Use the main agent as coordinator only. Keep manual lookup, codebase mapping, PL/HLS analysis, AIE graph/kernel analysis, host/runtime analysis, and data/verification analysis in separate focused subagents when two or more areas are involved.

Do not run Vitis, AIE, HLS, XRT, board, simulator, or project build commands locally unless the user explicitly asks for that exact command in the current turn.

## Workflow

1. Restate the failure or requested change in concrete terms.
2. Identify the affected domains:
   - `manual`: AMD reference lookup in `D:/科研/文章/references_md`
   - `aie`: `aie/Config.h`, `aie/TopGraph.*`, `aie/ProcessGraph/*`, `aie/ProcessUnit/*`
   - `pl`: `pl/TopPL.cpp`, `pl/Q0Ctrl.cpp`, `pl/*.cfg`
   - `host`: `ps/host.cpp`, XRT/adf runtime and profiling
   - `connectivity-build`: `conn.cfg`, `pl/*.cfg`, `Makefile`, PLIO names, stream_connect, CU names, generated artifact names
   - `data`: `data/gen_case.py`, `data/verify_srad.py`, generated txt formats
   - `build`: use `connectivity-build` unless the issue is purely generated-data naming, then coordinate with `data`
3. If two or more domains are independent, dispatch one subagent per domain. Give each subagent a narrow file list, exact search terms, and required output.
4. Synthesize subagent results into one patch plan. Separate facts from inferences.
5. Before editing AIE/PL/host project code, ask the user to confirm the exact files and changes.
6. After confirmed edits, do static self-check only. Do not claim compile/sim/board verification.

## Manual Lookup

Never read the four AMD manuals end to end. Use targeted `rg -n` queries against only the likely manuals:

```powershell
rg -n "<exact error text|API|concept>" "D:/科研/文章/references_md/UG1079.md" "D:/科研/文章/references_md/UG1399.md" "D:/科研/文章/references_md/UG1076.md" "D:/科研/文章/references_md/AM009.md"
```

Use this routing:

| Question | Search first |
| --- | --- |
| ADF graph, kernels, buffers, margins, PLIO, GMIO, pktstream, profiling | `UG1079.md` |
| Vitis linking, v++, system config, packaging, XRT host flow | `UG1399.md` |
| Versal architecture, AIE array, NoC, memory hierarchy | `AM009.md` |
| Platform, board, boot, runtime environment details | `UG1076.md` |

For paper or algorithm context, search only relevant files in the same directory such as `SRAD.md`, `StencilFlow...md`, `SPARTA...md`, or `CHARM.md`.

## Subagent Prompt Contract

Each subagent must return:

- `Scope`: files and manuals actually read
- `Finding`: concrete issue or confirmation
- `Evidence`: file:line references or manual section/query hits
- `Patch impact`: what should change, or why no change is needed
- `Risk`: what the remote build/sim/board run must validate

Subagents should not edit files unless the main agent explicitly delegates a confirmed edit to that single domain.

## Main Agent Synthesis

When results come back, produce:

- Minimal root-cause explanation
- One recommended fix path
- Alternative only if it is genuinely viable
- Exact file/function changes
- Remote validation commands for the user to run, without running them locally

If evidence conflicts, do not average the answers. Re-open only the conflicting files/manual sections and resolve the contradiction before recommending edits.

## Project Invariants

Preserve these unless the user explicitly changes the design:

- `ours_192lane` uses row-stream SRAD, not the old tile+halo model.
- Input row format is 256 data floats plus q0 plus pad: 258 floats.
- Output row format is 256 floats.
- PLIO width is `plio_64_bits` for row data unless the user asks to redesign connectivity.
- Keep Q0Ctrl feedback; do not replace q0 with s_axilite or RTP without confirmation.
- Keep multi-iteration ping-pong semantics.
- Treat `data/gen_case.py` and host CPU reference logic as numerical semantics authorities.

## Good Requests To Subagents

Manual lookup:

```text
You are the manual lookup subagent. Search only D:/科研/文章/references_md/UG1079.md and UG1399.md for: input_buffer margin inherited_extent dimensions fifo_depth pktstream plio_64_bits. Return only relevant line references and short conclusions for SRAD ours_192lane. Do not edit files.
```

AIE graph/kernel:

```text
You are the AIE graph/kernel subagent. Read aie/Config.h, aie/ProcessUnit/srad.h, aie/ProcessUnit/srad_local_q.cc, aie/ProcessUnit/srad_coeff_update.cc, and aie/ProcessGraph/StencilCoreGraph.h. Check row dimensions, margins, and K1/K2 delay alignment. Return findings with file:line references. Do not edit files.
```

PL/host/connectivity:

```text
You are the PL/host/connectivity subagent. Read pl/TopPL.cpp, pl/Q0Ctrl.cpp, conn.cfg, pl/*.cfg, and ps/host.cpp. Check stream names, CU names, q0/stat handshakes, and XRT launch order. Return likely mismatch points. Do not edit files.
```

Build/connectivity:

```text
You are the build/connectivity subagent. Read conn.cfg, pl/TopPL.cfg, pl/Q0Ctrl.cfg, Makefile, aie/TopGraph.h, aie/TopGraph.cpp, and ps/host.cpp only as needed for names. Check stream_connect lines, PLIO names, kernel/CU names, generated xclbin/package targets, PLIO data file names, and host kernel-opening names. Return exact mismatches with file:line references. Do not edit files.
```

Data verification:

```text
You are the data verification subagent. Read data/gen_case.py, data/verify_srad.py, Makefile, and the PLIO/data file names used by aie/TopGraph.*. Check whether generated file formats match graph and PL expectations. Do not edit files.
```
