#!/usr/bin/env python3

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_case
import verify_srad


class SimSemanticsTest(unittest.TestCase):
    def test_make_sim_uses_six_rows_plus_flush_per_iteration(self):
        cfg, _lambda_default, _bypass = verify_srad.read_config()

        self.assertEqual(cfg["kSimRows"], 6)
        self.assertEqual(cfg["kSimIterations"], cfg["kSradIterations"])
        self.assertEqual(
            cfg["kRowsPerIterSim"],
            cfg["kSimRows"] + cfg["kFlushRows"],
        )
        self.assertEqual(
            cfg["kGraphRowsSim"],
            cfg["kRowsPerIterSim"] * cfg["kSimIterations"],
        )

        expected_plio64_lines = (
            cfg["kGraphRowsSim"] * cfg["kRowPhysElems"] // 2
        )
        self.assertEqual(expected_plio64_lines, 512)

        image = gen_case.make_default_image()
        q0sqr = gen_case.compute_q0sqr(image[: cfg["kSimRows"] * cfg["kCols"]], cfg["kSimRows"])
        stream = gen_case.make_rowstream_input(
            image,
            q0sqr,
            cfg["kRowsPerIterSim"],
            cfg["kSimRows"],
        )
        self.assertEqual(
            len(stream) // 2,
            cfg["kRowsPerIterSim"] * cfg["kRowPhysElems"] // 2,
        )

    def test_graph_uses_single_input_plio_fanout_for_two_core_inputs(self):
        graph = (
            Path(__file__).resolve().parents[1]
            / "aie"
            / "ProcessGraph"
            / "StencilCoreGraph.h"
        ).read_text(encoding="utf-8")

        self.assertIn("static constexpr int kInputCount = 2", graph)
        self.assertIn("port<input> in[kInputCount]", graph)
        self.assertIn("input_plio in_j", graph)
        self.assertIn("connect<>(in_j.out[0], core.in[0])", graph)
        self.assertIn("connect<>(in_j.out[0], core.in[1])", graph)
        self.assertIn("connect<>(in[0], k_local_q.in[0])", graph)
        self.assertIn("connect<>(in[1], k_coeff_update.in[1])", graph)
        self.assertNotIn("input_plio in_j_local", graph)
        self.assertNotIn("input_plio in_j_update", graph)

    def test_coeff_formula_uses_q0sqr_squared_in_denominator(self):
        q0sqr = 0.3323380007947052
        jc = 1.5284111271593896
        d_n = 0.8875575203260992
        d_s = -1.9865344469004076
        d_w = 1.7531147977500554
        d_e = 1.7875839306984802

        got = verify_srad.compute_c(jc, d_n, d_s, d_w, d_e, q0sqr, False)

        b = d_n + d_s + d_w + d_e
        dq = (jc + 0.25 * b) ** 2
        nq = 0.5 * (d_n * d_n + d_s * d_s + d_w * d_w + d_e * d_e)
        nq -= (1.0 / 16.0) * b * b
        expected = q0sqr * (1.0 + q0sqr) * dq
        expected /= nq + q0sqr * q0sqr * dq
        expected = verify_srad.clamp01(expected)

        self.assertAlmostEqual(got, expected)

    def test_aie_graph_uses_three_kernel_pipeline_with_dedicated_stats_kernel(self):
        graph = (
            Path(__file__).resolve().parents[1]
            / "aie"
            / "ProcessGraph"
            / "StencilCoreGraph.h"
        ).read_text(encoding="utf-8")
        makefile = (
            Path(__file__).resolve().parents[1] / "Makefile"
        ).read_text(encoding="utf-8")

        self.assertIn("kernel k_local_q", graph)
        self.assertIn("kernel k_coeff_update", graph)
        self.assertIn("kernel k_row_stats", graph)
        self.assertIn("kernel::create(srad_local_q)", graph)
        self.assertIn("kernel::create(srad_coeff_update)", graph)
        self.assertIn("kernel::create(srad_row_stats)", graph)
        self.assertIn("connect<>(k_coeff_update.out[0], k_row_stats.in[0])", graph)
        self.assertIn("connect<>(k_row_stats.out[0], out_j_next)", graph)
        self.assertIn(
            "dimensions(k_coeff_update.out[0]) = {srad_cfg::kUpdatedRowPhysElems}",
            graph,
        )
        self.assertIn(
            "dimensions(k_row_stats.out[0]) = {srad_cfg::kStatsRowPhysElems}",
            graph,
        )
        self.assertNotIn("srad_row_update", graph)
        self.assertNotIn("srad_row_update.cc", makefile)
        self.assertIn("./aie/ProcessUnit/srad_row_stats.cc", makefile)

    def test_make_sim_runs_correctness_compare_without_profile_outputs(self):
        makefile = (
            Path(__file__).resolve().parents[1] / "Makefile"
        ).read_text(encoding="utf-8")

        self.assertIn("python $(DATA_VERIFY) --compare", makefile)
        self.assertNotIn("--profile", makefile)
        self.assertNotIn("-wdb", makefile)
        self.assertNotIn("--online", makefile)

    def test_aie_sim_main_runs_every_generated_stream_row(self):
        top_graph = (
            Path(__file__).resolve().parents[1] / "aie" / "TopGraph.cpp"
        ).read_text(encoding="utf-8")

        self.assertIn("graphOursPLQ0.run(srad_cfg::kGraphRowsSim)", top_graph)
        self.assertNotIn("graphOursPLQ0.run(6)", top_graph)

    def test_board_host_runs_graph_for_finite_row_count(self):
        host = (
            Path(__file__).resolve().parents[1] / "ps" / "host.cpp"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "const int graph_rows = srad_cfg::kBoardGraphRowsPerIteration * active_iters",
            host,
        )
        self.assertIn("graphOursPLQ0.run(graph_rows)", host)
        self.assertIn("graphOursPLQ0.wait()", host)
        self.assertIn("graphOursPLQ0.end()", host)
        self.assertNotIn("graphOursPLQ0.run();", host)

    def test_board_host_has_parallel_wait_diagnostics(self):
        host = (
            Path(__file__).resolve().parents[1] / "ps" / "host.cpp"
        ).read_text(encoding="utf-8")

        self.assertIn("#include <atomic>", host)
        self.assertIn("#include <thread>", host)
        self.assertIn("std::thread graph_wait_thread", host)
        self.assertIn("std::thread toppl_wait_thread", host)
        self.assertIn("[diag] wait_status", host)
        self.assertIn("toppl=%s%s graph=%s%s", host)
        self.assertNotIn("q0_status=", host)
        self.assertNotIn("q0_stat_phase=", host)
        self.assertNotIn("q0_send_phase=", host)

    def test_board_path_uses_single_toppl_local_q0(self):
        root = Path(__file__).resolve().parents[1]
        toppl = (root / "pl" / "TopPL.cpp").read_text(encoding="utf-8")
        host = (root / "ps" / "host.cpp").read_text(encoding="utf-8")
        conn = (root / "conn.cfg").read_text(encoding="utf-8")
        makefile = (root / "Makefile").read_text(encoding="utf-8")

        self.assertIn("float compute_q0sqr_from_sums", toppl)
        self.assertIn("const float q0sqr = compute_q0sqr_from_sums(sum, sum2);", toppl)
        self.assertIn("if ((iter & 1) == 0)", toppl)
        self.assertIn("run_all_strips(image,", toppl)
        self.assertIn("run_all_strips(output,", toppl)
        self.assertNotIn("stat_to_q0.write", toppl)
        self.assertNotIn("q0_from_ctrl.read", toppl)
        self.assertNotIn("auto q0_run", host)
        self.assertNotIn("open_q0_kernel", host)
        self.assertNotIn("Q0Ctrl", conn)
        self.assertNotIn("stat_to_q0", conn)
        self.assertNotIn("q0_from_ctrl", conn)
        self.assertNotIn("Q0_KERNEL", makefile)
        self.assertIn("rm -rf _x", makefile)
        self.assertIn("rm -f ./pl/Q0Ctrl.xo", makefile)

    def test_mid_buffer_uses_aligned_physical_row_stride(self):
        cfg, _lambda_default, _bypass = verify_srad.read_config()
        config = (
            Path(__file__).resolve().parents[1] / "aie" / "Config.h"
        ).read_text(encoding="utf-8")
        local_q = (
            Path(__file__).resolve().parents[1]
            / "aie"
            / "ProcessUnit"
            / "srad_local_q.cc"
        ).read_text(encoding="utf-8")
        coeff_update = (
            Path(__file__).resolve().parents[1]
            / "aie"
            / "ProcessUnit"
            / "srad_coeff_update.cc"
        ).read_text(encoding="utf-8")

        self.assertEqual(cfg["kMidPlaneStride"], cfg["kRowPhysElems"])
        self.assertEqual(cfg["kMidRecordElems"], 4)
        self.assertEqual(
            cfg["kMidElemsPerRow"],
            cfg["kMidRecordElems"] * cfg["kMidPlaneStride"],
        )
        self.assertEqual((cfg["kMidElemsPerRow"] * 4) % 16, 0)
        self.assertIn("kMidBytesPerRow % 16", config)
        self.assertIn(
            "kCenterQ2Plane * srad_cfg::kMidPlaneStride",
            local_q,
        )
        self.assertIn(
            "kCenterQ2Plane * srad_cfg::kMidPlaneStride",
            coeff_update,
        )

    def test_local_q_outputs_q2_and_update_computes_coefficients(self):
        root = Path(__file__).resolve().parents[1]
        local_q = (
            root / "aie" / "ProcessUnit" / "srad_local_q.cc"
        ).read_text(encoding="utf-8")
        coeff_update = (
            root / "aie" / "ProcessUnit" / "srad_coeff_update.cc"
        ).read_text(encoding="utf-8")

        self.assertIn("inline void compute_q2_vec(", local_q)
        self.assertIn("*q2_out = q2_v.to_native();", local_q)
        self.assertIn("*valid_out = valid_v.to_native();", local_q)
        self.assertIn("center_q2", local_q)
        self.assertIn("south_q2", local_q)
        self.assertNotIn("encode_coeff_vec", local_q)
        self.assertNotIn("num / den", local_q)
        self.assertNotIn("clamp01(num", local_q)
        self.assertIn("inline v8float compute_coeff_vec(", coeff_update)
        self.assertIn("v8float q2", coeff_update)
        self.assertIn("fpmac(q2, q0sqr", coeff_update)
        self.assertIn("aie::inv(safe_coeff_den)", coeff_update)
        self.assertIn("aie::select(one_v, clamped, q0_pos_mask)", coeff_update)

    def test_mid_buffer_reuses_center_plane_for_east_coefficients(self):
        root = Path(__file__).resolve().parents[1]
        local_q = (
            root / "aie" / "ProcessUnit" / "srad_local_q.cc"
        ).read_text(encoding="utf-8")
        coeff_update = (
            root / "aie" / "ProcessUnit" / "srad_coeff_update.cc"
        ).read_text(encoding="utf-8")

        self.assertNotIn("kEastValuePlane", local_q)
        self.assertNotIn("kEastTagPlane", local_q)
        self.assertNotIn("east_value", local_q)
        self.assertNotIn("east_tag", local_q)
        self.assertNotIn("c_east", local_q)
        self.assertNotIn("kEastValuePlane", coeff_update)
        self.assertNotIn("kEastTagPlane", coeff_update)
        self.assertNotIn("east_value", coeff_update)
        self.assertNotIn("east_tag", coeff_update)
        self.assertIn("constexpr int kMidRecordElems = 4;", (
            root / "aie" / "Config.h"
        ).read_text(encoding="utf-8"))
        self.assertIn(
            "store_vec(center_q2, col,\n"
            "                  select_data_lanes(center_q2_vec, data_mask, one_v));",
            local_q,
        )
        self.assertIn(
            "store_vec(center_valid, col,\n"
            "                  select_data_lanes(center_valid_vec, data_mask, zero_v));",
            local_q,
        )
        self.assertNotIn("fill_plane_padding(", local_q)
        self.assertIn("const int next_col =", coeff_update)
        self.assertIn("(chunk + 1) & (kChunksPerRow - 1)", coeff_update)
        self.assertIn("load_vec(row_m, next_col)", coeff_update)
        self.assertIn("shift_right_with_zero(coeff, coeff_next, one)", coeff_update)
        self.assertNotIn("load_vec(row_m, col + srad_cfg::kLanes)", coeff_update)

    def test_local_q_and_update_hoist_q0_and_split_hot_paths(self):
        root = Path(__file__).resolve().parents[1]
        local_q = (
            root / "aie" / "ProcessUnit" / "srad_local_q.cc"
        ).read_text(encoding="utf-8")
        coeff_update = (
            root / "aie" / "ProcessUnit" / "srad_coeff_update.cc"
        ).read_text(encoding="utf-8")

        self.assertNotIn("q0sqr", local_q)
        self.assertIn(
            "const float q0_den_scalar = q0sqr * (srad_math::kOne + q0sqr);",
            coeff_update,
        )
        self.assertIn("const v8float q0sqr_vec = splat(q0sqr);", coeff_update)
        self.assertIn("const v8float q0_den = splat(q0_den_scalar);", coeff_update)
        self.assertIn("const aie::vector<float, srad_cfg::kLanes> q0_den_v(q0_den);", coeff_update)
        self.assertIn("compute_q2_vec(", local_q)
        self.assertIn("compute_coeff_vec(", coeff_update)
        self.assertNotIn("encode_coeff_interior(", local_q)
        self.assertNotIn("encode_coeff_boundary(", local_q)
        self.assertNotIn("read_row_value(", local_q)
        self.assertRegex(
            local_q,
            r"for \(int chunk = 0; chunk < kChunksPerRow; \+\+chunk\)\s+"
            r"chess_prepare_for_pipelining\s+"
            r"chess_loop_range\(kChunksPerRow, kChunksPerRow\) \{",
        )
        self.assertIn("const float* __restrict base = in_j.data();", local_q)
        self.assertIn("float* __restrict mid = out_q2.data();", local_q)

        self.assertIn("const float* __restrict mid = in_q2.data();", coeff_update)
        self.assertIn("const float* __restrict base = in_j.data();", coeff_update)
        self.assertIn("float* __restrict out = out_j_next.data();", coeff_update)
        self.assertIn("chess_prepare_for_pipelining", coeff_update)
        self.assertIn("chess_loop_range", coeff_update)

    def test_local_q_uses_vector_q2_compute_with_valid_lane_mask(self):
        root = Path(__file__).resolve().parents[1]
        local_q = (
            root / "aie" / "ProcessUnit" / "srad_local_q.cc"
        ).read_text(encoding="utf-8")

        self.assertNotIn("#include <aie_api/aie.hpp>", local_q)
        self.assertIn("inline void compute_q2_vec(", local_q)
        self.assertIn("#define SRAD_ENABLE_AIE_VEC_HELPERS 1", local_q)
        self.assertIn('#include "ProcessUnit/srad.h"', local_q)
        srad_h = (root / "aie" / "ProcessUnit" / "srad.h").read_text(
            encoding="utf-8"
        )
        self.assertIn("namespace srad_vec", srad_h)
        self.assertIn("#include <aie_api/aie.hpp>", srad_h)
        self.assertIn("inline v8float splat(", srad_h)
        self.assertIn("inline aie::mask<srad_cfg::kLanes> make_data_lane_mask(", srad_h)
        self.assertIn("inline v8float select_data_lanes(", srad_h)
        self.assertNotIn("inline v8float mask_data_lanes(", local_q)
        self.assertIn("const auto data_mask = make_data_lane_mask(col);", local_q)
        self.assertIn("const auto next_data_mask = make_data_lane_mask(next_col);", local_q)
        self.assertIn("const int col = chunk * srad_cfg::kLanes;", local_q)
        self.assertIn("const int next_col =", local_q)
        self.assertIn("const v8float center_c = select_data_lanes(", local_q)
        self.assertIn("const v8float south_c = center_s;", local_q)
        self.assertIn("compute_q2_vec(", local_q)
        self.assertIn("shift_left_with_zero(center_c, prev_center_c, one)", local_q)
        self.assertIn("shift_right_with_zero(center_c, center_next, one)", local_q)
        self.assertIn(
            "store_vec(south_q2, col,\n"
            "                  select_data_lanes(south_q2_vec, data_mask, zero_v));",
            local_q,
        )
        self.assertNotIn("fill_plane_padding(", local_q)

    def test_coeff_update_uses_vector_coeff_compute_with_valid_lane_mask(self):
        root = Path(__file__).resolve().parents[1]
        coeff_update = (
            root / "aie" / "ProcessUnit" / "srad_coeff_update.cc"
        ).read_text(encoding="utf-8")
        row_stats = (
            root / "aie" / "ProcessUnit" / "srad_row_stats.cc"
        ).read_text(encoding="utf-8")

        self.assertNotIn("#include <aie_api/aie.hpp>", coeff_update)
        self.assertIn("inline v8float compute_coeff_vec(", coeff_update)
        self.assertIn("aie::inv(safe_q0_den)", coeff_update)
        self.assertIn("aie::inv(safe_coeff_den)", coeff_update)
        self.assertIn("aie::select(one_v, clamped, q0_pos_mask)", coeff_update)
        self.assertIn("#define SRAD_ENABLE_AIE_VEC_HELPERS 1", coeff_update)
        srad_h = (root / "aie" / "ProcessUnit" / "srad.h").read_text(
            encoding="utf-8"
        )
        self.assertIn("namespace srad_vec", srad_h)
        self.assertIn("#include <aie_api/aie.hpp>", srad_h)
        self.assertIn("inline v8float splat(", srad_h)
        self.assertIn("inline aie::mask<srad_cfg::kLanes> make_data_lane_mask(", srad_h)
        self.assertIn("inline v8float select_data_lanes(", srad_h)
        self.assertNotIn("inline v8float mask_data_lanes(", coeff_update)
        self.assertIn("const auto data_mask = make_data_lane_mask(col);", coeff_update)
        self.assertIn("const auto next_data_mask = make_data_lane_mask(next_col);", coeff_update)
        self.assertRegex(
            coeff_update,
            r"for \(int chunk = 0; chunk < kChunksPerRow; \+\+chunk\)\s+"
            r"chess_prepare_for_pipelining\s+"
            r"chess_loop_range\(kChunksPerRow, kChunksPerRow\) \{",
        )
        self.assertIn("const int col = chunk * srad_cfg::kLanes;", coeff_update)
        self.assertIn("const v8float raw_jc = load_vec(row_m, col);", coeff_update)
        self.assertIn("const v8float jc = select_data_lanes(raw_jc, data_mask, zero_v);", coeff_update)
        self.assertIn("const int next_col =", coeff_update)
        self.assertIn("const v8float next_row_m =", coeff_update)
        self.assertIn(
            "select_data_lanes(raw_next_jc, next_data_mask, zero_v);",
            coeff_update,
        )
        self.assertIn("shift_right_with_zero(jc, next_row_m, one)", coeff_update)
        self.assertIn("const v8float next_masked = select_data_lanes(next, data_mask, zero_v);", coeff_update)
        self.assertIn("store_vec(out, col, next_masked);", coeff_update)
        self.assertNotIn("accumulate_stats", coeff_update)
        self.assertNotIn("out[srad_cfg::kStatSumPadIndex]", coeff_update)
        self.assertNotIn("out[srad_cfg::kStatSum2PadIndex]", coeff_update)
        self.assertIn("void srad_row_stats(", row_stats)
        self.assertIn("accumulate_stats(next_masked, row_sum, row_sum2);", row_stats)
        self.assertIn("out[srad_cfg::kStatSumPadIndex] = row_sum;", row_stats)
        self.assertIn("out[srad_cfg::kStatSum2PadIndex] = row_sum2;", row_stats)
        self.assertNotIn(
            "for (int col = 1; col < srad_cfg::kRowDataElems - 1; ++col)",
            coeff_update,
        )

    def test_toppl_decouples_ddr_and_aie_streams_with_local_fifos(self):
        toppl = (
            Path(__file__).resolve().parents[1] / "pl" / "TopPL.cpp"
        ).read_text(encoding="utf-8")

        self.assertIn("hls::stream<plio_word_t> to_aie_words;", toppl)
        self.assertIn("hls::stream<plio_word_t> from_aie_words;", toppl)
        self.assertIn("#pragma HLS STREAM variable=to_aie_words depth=128", toppl)
        self.assertIn("#pragma HLS STREAM variable=from_aie_words depth=128", toppl)
        self.assertRegex(
            toppl,
            r"#pragma HLS DATAFLOW\s+"
            r"prepare_input_rows\(current, strip, q0sqr, to_aie_words\);\s+"
            r"forward_input_words\(to_aie_words, out_j\);\s+"
            r"capture_output_words\(in_j_next, from_aie_words\);\s+"
            r"store_output_rows\(next, strip, next_sum, next_sum2, from_aie_words\);",
        )

    def test_toppl_avoids_hls_unsupported_m_axi_pointer_selection(self):
        toppl = (
            Path(__file__).resolve().parents[1] / "pl" / "TopPL.cpp"
        ).read_text(encoding="utf-8")

        self.assertNotIn("const float* current = (iter & 1) ? output : image;", toppl)
        self.assertNotIn("float* next = (iter & 1) ? image : output;", toppl)
        self.assertIn("run_all_strips(image,", toppl)
        self.assertIn("run_all_strips(output,", toppl)

    def test_board_config_keeps_aie_width_but_adds_4000x4000_strip_volume(self):
        cfg, _lambda_default, _bypass = verify_srad.read_config()

        self.assertEqual(cfg["kRows"], 125)
        self.assertEqual(cfg["kCols"], 125)
        self.assertEqual(cfg["kRowPhysElems"], 128)
        self.assertEqual(cfg["kBoardRows"], 4000)
        self.assertEqual(cfg["kBoardCols"], 4000)
        self.assertEqual(cfg["kBoardIterations"], 100)
        self.assertEqual(cfg["kBoardStrips"], 32)
        self.assertEqual(cfg["kBoardRowsPerStrip"], 4002)
        self.assertEqual(
            cfg["kBoardGraphRowsPerIteration"],
            cfg["kBoardStrips"] * cfg["kBoardRowsPerStrip"],
        )

    def test_data_generation_defaults_to_board_input_and_gates_sim_data(self):
        gen_case_text = (
            Path(__file__).resolve().parents[1] / "data" / "gen_case.py"
        ).read_text(encoding="utf-8")
        verify_text = (
            Path(__file__).resolve().parents[1] / "data" / "verify_srad.py"
        ).read_text(encoding="utf-8")
        makefile = (
            Path(__file__).resolve().parents[1] / "Makefile"
        ).read_text(encoding="utf-8")

        self.assertIn('write_default_image_matrix(base / "input_image.txt"', gen_case_text)
        self.assertIn('parser.add_argument(', gen_case_text)
        self.assertIn('"--sim"', gen_case_text)
        self.assertIn("if not args.sim:", gen_case_text)
        self.assertIn('write_matrix(base / "input_image_sim.txt"', gen_case_text)
        self.assertIn('default=base / "input_image_sim.txt"', verify_text)
        self.assertIn("python $(DATA_GEN)", makefile)
        self.assertIn("python $(DATA_GEN) --sim", makefile)
        self.assertNotIn("data: $(GENERATED_DATA_FILES)", makefile)

    def test_makefile_has_fast_x86sim_correctness_entrypoint(self):
        makefile = (
            Path(__file__).resolve().parents[1] / "Makefile"
        ).read_text(encoding="utf-8")

        self.assertIn(".PHONY: clean all kernels aie sim sim_fast", makefile)
        self.assertIn("TARGET_ORIGIN := $(origin TARGET)", makefile)
        self.assertIn("ifeq ($(TARGET_ORIGIN),undefined)", makefile)
        self.assertIn("AIE_TARGET_STAMP := .aie_target.$(TARGET)", makefile)
        self.assertIn("$(AIE_TARGET_STAMP):", makefile)
        self.assertIn("rm -rf Work $(GRAPH_O)", makefile)
        self.assertIn("rm -f .aie_target.*", makefile)
        self.assertIn("sim_fast:", makefile)
        self.assertIn("$(MAKE) sim TARGET=sw_emu", makefile)

    def test_make_sim_removes_stale_sim_outputs_before_compare(self):
        makefile = (
            Path(__file__).resolve().parents[1] / "Makefile"
        ).read_text(encoding="utf-8")

        self.assertIn("SIM_OUTPUT := ./data/aiesim_j_next.txt", makefile)
        self.assertIn("SIM_OUTPUT_DIRS :=", makefile)
        self.assertIn("rm -f $(SIM_OUTPUT)", makefile)
        self.assertIn("rm -rf $(SIM_OUTPUT_DIRS)", makefile)

    def test_verify_resolves_x86simulator_output_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requested = root / "data" / "aiesim_j_next.txt"
            produced = root / "x86simulator_output" / "data" / "aiesim_j_next.txt"
            produced.parent.mkdir(parents=True)
            produced.write_text("1.0 2.0\n", encoding="utf-8")

            self.assertEqual(
                verify_srad.resolve_aie_output_path(requested),
                produced,
            )

    def test_verify_resolves_work_x86simulator_output_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requested = root / "data" / "aiesim_j_next.txt"
            produced = (
                root
                / "Work"
                / "x86simulator_output"
                / "data"
                / "aiesim_j_next.txt"
            )
            produced.parent.mkdir(parents=True)
            produced.write_text("1.0 2.0\n", encoding="utf-8")

            self.assertEqual(
                verify_srad.resolve_aie_output_path(requested),
                produced,
            )

    def test_verify_reports_checked_output_locations_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            requested = Path(tmp) / "data" / "aiesim_j_next.txt"

            with self.assertRaises(FileNotFoundError) as ctx:
                verify_srad.resolve_aie_output_path(requested)

            message = str(ctx.exception)
            self.assertIn("AIE output not found", message)
            self.assertIn("x86simulator_output", message)


if __name__ == "__main__":
    unittest.main()
