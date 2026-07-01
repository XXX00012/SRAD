#!/usr/bin/env python3

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_case
import verify_srad


ROOT = Path(__file__).resolve().parents[1]


class CudaStyleAieContractTest(unittest.TestCase):
    def test_config_keeps_cuda_block_contract(self):
        cfg, lambda_default = verify_srad.read_config()

        self.assertEqual(cfg["kCudaBlockElems"], 512)
        self.assertEqual(cfg["kReducePacketElems"], cfg["kMetaPacketElems"] + 2 * cfg["kCudaBlockElems"])
        self.assertEqual(cfg["kCoeffInputElems"], 5 * cfg["kCudaBlockElems"])
        self.assertEqual(cfg["kUpdateInputElems"], 8 * cfg["kCudaBlockElems"])
        self.assertEqual(cfg["kReduceOutputElems"], 2)
        self.assertEqual(lambda_default, 0.5)

    def test_aie_top_graph_uses_four_cuda_stage_graphs(self):
        graph = (ROOT / "aie" / "TopGraph.cpp").read_text(encoding="utf-8")
        header = (
            ROOT / "aie" / "ProcessGraph" / "StencilCoreGraph.h"
        ).read_text(encoding="utf-8")

        self.assertIn('GraphPrepare graphPrepare("gpu_prepare");', graph)
        self.assertIn('GraphReduce graphReduce("gpu_reduce");', graph)
        self.assertIn('GraphSradCoeff graphSradCoeff("gpu_srad");', graph)
        self.assertIn('GraphSradUpdate graphSradUpdate("gpu_srad2");', graph)
        self.assertIn("graphPrepare.run(1);", graph)
        self.assertIn("graphReduce.run(1);", graph)
        self.assertIn("graphSradCoeff.run(1);", graph)
        self.assertIn("graphSradUpdate.run(1);", graph)

        for name in (
            "gpu_prepare_i.txt",
            "gpu_prepare_sums.txt",
            "gpu_prepare_sums2.txt",
            "gpu_reduce_packet.txt",
            "gpu_reduce_partial.txt",
            "gpu_srad_neighbors.txt",
            "gpu_srad_q0.txt",
            "gpu_srad_dN.txt",
            "gpu_srad_dS.txt",
            "gpu_srad_dW.txt",
            "gpu_srad_dE.txt",
            "gpu_srad_c.txt",
            "gpu_srad2_update.txt",
            "gpu_srad2_meta.txt",
            "gpu_srad2_i_next.txt",
        ):
            self.assertIn(name, header)

        self.assertNotIn("GraphOursPLQ0", graph + header)
        self.assertNotIn("srad_local_q", graph + header)
        self.assertNotIn("srad_coeff_update", graph + header)

    def test_makefile_builds_four_hls_bridge_kernels(self):
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("PL_KERNELS := PreparePL ReducePL CoeffPL UpdatePL", makefile)
        self.assertIn("KERNEL_XOS := $(addprefix ./pl/,$(addsuffix .xo,$(PL_KERNELS)))", makefile)
        self.assertIn("./pl/%.xo:", makefile)
        self.assertIn("-k $*", makefile)
        self.assertIn("./aie/ProcessUnit/srad_prepare.cc", makefile)
        self.assertIn("./aie/ProcessUnit/srad_reduce.cc", makefile)
        self.assertIn("./aie/ProcessUnit/srad_coeff.cc", makefile)
        self.assertIn("./aie/ProcessUnit/srad_update.cc", makefile)
        self.assertNotIn("srad_local_q.cc", makefile)
        self.assertNotIn("srad_coeff_update.cc", makefile)

    def test_connectivity_matches_four_graph_plios(self):
        conn = (ROOT / "conn.cfg").read_text(encoding="utf-8")

        for kernel in ("PreparePL", "ReducePL", "CoeffPL", "UpdatePL"):
            self.assertIn(f"nk={kernel}:1:{kernel}_0", conn)

        for line in (
            "stream_connect=PreparePL_0.out_i:ai_engine_0.gpu_prepare_in_i:256",
            "stream_connect=ai_engine_0.gpu_prepare_out_sums:PreparePL_0.in_sums:256",
            "stream_connect=ai_engine_0.gpu_prepare_out_sums2:PreparePL_0.in_sums2:256",
            "stream_connect=ReducePL_0.out_packet:ai_engine_0.gpu_reduce_in_packet:256",
            "stream_connect=ai_engine_0.gpu_reduce_out_partial:ReducePL_0.in_partial:256",
            "stream_connect=CoeffPL_0.out_neighbors:ai_engine_0.gpu_srad_in_neighbors:256",
            "stream_connect=CoeffPL_0.out_q0:ai_engine_0.gpu_srad_in_q0:256",
            "stream_connect=ai_engine_0.gpu_srad_out_dN:CoeffPL_0.in_dN:256",
            "stream_connect=ai_engine_0.gpu_srad_out_dS:CoeffPL_0.in_dS:256",
            "stream_connect=ai_engine_0.gpu_srad_out_dW:CoeffPL_0.in_dW:256",
            "stream_connect=ai_engine_0.gpu_srad_out_dE:CoeffPL_0.in_dE:256",
            "stream_connect=ai_engine_0.gpu_srad_out_c:CoeffPL_0.in_c:256",
            "stream_connect=UpdatePL_0.out_update:ai_engine_0.gpu_srad2_in_update:256",
            "stream_connect=UpdatePL_0.out_meta:ai_engine_0.gpu_srad2_in_meta:256",
            "stream_connect=ai_engine_0.gpu_srad2_out_i_next:UpdatePL_0.in_i_next:256",
        ):
            self.assertIn(line, conn)

        self.assertNotIn("ours_plq0", conn)
        self.assertNotIn("TopPL_0", conn)
        self.assertNotIn("Q0Ctrl", conn)

    def test_hls_bridge_kernels_match_aie_stage_contract(self):
        toppl = (ROOT / "pl" / "TopPL.cpp").read_text(encoding="utf-8")

        for fn in ("PreparePL", "ReducePL", "CoeffPL", "UpdatePL"):
            self.assertIn(f"void {fn}(", toppl)

        self.assertIn("hls::stream<plio_word_t>& out_i", toppl)
        self.assertIn("hls::stream<plio_word_t>& in_sums", toppl)
        self.assertIn("hls::stream<plio_word_t>& out_packet", toppl)
        self.assertIn("hls::stream<plio_word_t>& out_neighbors", toppl)
        self.assertIn("hls::stream<plio_word_t>& out_q0", toppl)
        self.assertIn("hls::stream<plio_word_t>& out_update", toppl)
        self.assertIn("hls::stream<plio_word_t>& out_meta", toppl)
        self.assertIn("hls::stream<plio_word_t>& in_i_next", toppl)
        self.assertIn("srad_cfg::kCoeffInputPlanes", toppl)
        self.assertIn("srad_cfg::kUpdateInputPlanes", toppl)
        self.assertNotIn("out_j", toppl)
        self.assertNotIn("in_j_next", toppl)

    def test_host_runs_prepare_reduce_coeff_update_graphs(self):
        host = (ROOT / "ps" / "host.cpp").read_text(encoding="utf-8")

        for graph in ("graphPrepare", "graphReduce", "graphSradCoeff", "graphSradUpdate"):
            self.assertIn(f"{graph}.init();", host)
            self.assertIn(f"{graph}.end();", host)

        self.assertIn('open_kernel(device, xrt_uuid, "PreparePL")', host)
        self.assertIn('open_kernel(device, xrt_uuid, "ReducePL")', host)
        self.assertIn('open_kernel(device, xrt_uuid, "CoeffPL")', host)
        self.assertIn('open_kernel(device, xrt_uuid, "UpdatePL")', host)
        self.assertIn("const float q0sqr = compute_q0sqr(partials_map, blocks);", host)
        self.assertIn("std::swap(current_bo, next_bo);", host)
        self.assertNotIn("graphOursPLQ0", host)
        self.assertNotIn("TopPL", host)

    def test_data_generation_uses_gpu_plio_names(self):
        gen_case_text = (ROOT / "data" / "gen_case.py").read_text(encoding="utf-8")
        verify_text = (ROOT / "data" / "verify_srad.py").read_text(encoding="utf-8")

        for name in (
            "gpu_prepare_i.txt",
            "gpu_reduce_packet.txt",
            "gpu_srad_neighbors.txt",
            "gpu_srad_q0.txt",
            "gpu_srad2_update.txt",
            "gpu_srad2_meta.txt",
            "gold_gpu_srad2_i_next.txt",
        ):
            self.assertIn(name, gen_case_text)

        self.assertIn('("gpu_prepare_sums.txt", "gold_gpu_prepare_sums.txt")', verify_text)
        self.assertIn('("gpu_srad2_i_next.txt", "gold_gpu_srad2_i_next.txt")', verify_text)
        self.assertNotIn("plio_ours_j_tile", verify_text)
        self.assertNotIn("gold_srad_rowstream", verify_text)

    def test_verify_resolves_x86simulator_output_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requested = root / "data" / "gpu_srad2_i_next.txt"
            produced = root / "x86simulator_output" / "data" / "gpu_srad2_i_next.txt"
            produced.parent.mkdir(parents=True)
            produced.write_text("1.0 2.0\n", encoding="utf-8")

            self.assertEqual(verify_srad.resolve_output_path(requested), produced)


if __name__ == "__main__":
    unittest.main()
