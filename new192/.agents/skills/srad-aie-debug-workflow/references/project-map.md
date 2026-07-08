# ours_192lane Project Map

Use this map to reduce repeated discovery.

## Core Files

| Area | Files |
| --- | --- |
| AIE config | `aie/Config.h` |
| AIE graph | `aie/TopGraph.h`, `aie/TopGraph.cpp`, `aie/ProcessGraph/StencilCoreGraph.h`, `aie/ProcessGraph/StencilCoreGraph.cpp` |
| AIE kernels | `aie/ProcessUnit/srad.h`, `aie/ProcessUnit/include.h`, `aie/ProcessUnit/srad_local_q.cc`, `aie/ProcessUnit/srad_coeff_update.cc` |
| PL kernels | `pl/TopPL.cpp`, `pl/Q0Ctrl.cpp`, `pl/TopPL.cfg`, `pl/Q0Ctrl.cfg` |
| Host | `ps/host.cpp` |
| Connectivity | `conn.cfg` |
| Data | `data/gen_case.py`, `data/verify_srad.py`, `data/test_sim_semantics.py` |
| Build/package | `Makefile`, `run.sh`, `xrt.ini`, `systemConfig.h` |

## Local References

Manual directory: `D:/科研/文章/references_md`

Primary AMD manuals:

- `UG1079.md`: AI Engine graph/kernel/programming model, buffers, PLIO, profiling.
- `UG1399.md`: Vitis acceleration flow, linking, packaging, host/XRT flow.
- `AM009.md`: Versal architecture and AIE array context.
- `UG1076.md`: platform and runtime environment context.

Relevant papers:

- `SRAD.md`
- `StencilFlow_Mapping_Large_Stencil_Programs_to_Distributed_Spatial_Computing_Systems.md`
- `Singh 等 - 2023 - SPARTA Spatial Acceleration for Efficient and Scalable Horizontal Diffusion Weather Stencil Computa.md`
- `CHARM.md`

## Static Search Commands

Use these before reading large files:

```powershell
rg -n "margin|input_buffer|dimensions|fifo_depth|pktstream|plio_64_bits|connect<|graph.run" aie
rg -n "q0|stat|hls::stream|DATAFLOW|INTERFACE|axis|m_axi|worker|pack_two_floats" pl ps conn.cfg
rg -n "PLIO|input_plio|output_plio|kRowElems|kOutputRowElems|kSimRows|kSradIterations" aie data Makefile
```

## Do Not Run Locally

Avoid local execution of:

- `make`, `v++`, `aiecompiler`, `aiesimulator`
- `xsim`, `hw_emu`, board/XRT execution
- environment setup scripts such as Vitis/XRT settings

The user runs these remotely and reports logs back.
