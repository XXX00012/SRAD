#!/usr/bin/env python3

import sys
import tempfile
import unittest
import re
import struct
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

    def test_graph_uses_parallel_input_plios_with_per_lane_fanout(self):
        graph = (
            Path(__file__).resolve().parents[1]
            / "aie"
            / "ProcessGraph"
            / "StencilCoreGraph.h"
        ).read_text(encoding="utf-8")

        self.assertIn("static constexpr int kInputCount = 2", graph)
        self.assertIn("port<input> in[kInputCount]", graph)
        self.assertIn("static constexpr int kNumLanes = srad_cfg::kParallelLanes", graph)
        self.assertIn("SradCoreGraph core[kNumLanes]", graph)
        self.assertIn("input_plio in_j[kNumLanes]", graph)
        self.assertIn("connect<>(in_j[lane].out[0], core[lane].in[0])", graph)
        self.assertIn("connect<>(in_j[lane].out[0], core[lane].in[1])", graph)
        self.assertNotIn("location<PLIO>", graph)
        self.assertNotIn("shim(srad_cfg::lane_tile_col", graph)
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

    def test_aie_graph_uses_two_kernel_pipeline_without_row_update_kernel(self):
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
        self.assertIn("kernel::create(srad_local_q)", graph)
        self.assertIn("kernel::create(srad_coeff_update)", graph)
        self.assertNotIn("srad_row_update", graph)
        self.assertNotIn("srad_row_update.cc", makefile)

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

    def test_board_host_waits_for_twelve_toppl_q0ctrl_and_graph(self):
        host = (
            Path(__file__).resolve().parents[1] / "ps" / "host.cpp"
        ).read_text(encoding="utf-8")

        self.assertIn("std::vector<xrt::kernel> toppl;", host)
        self.assertIn("toppl.reserve(srad_cfg::kTopPlWorkers);", host)
        self.assertIn("for (int worker = 0; worker < srad_cfg::kTopPlWorkers; ++worker)", host)
        self.assertIn('"TopPL_" + std::to_string(worker)', host)
        self.assertIn("toppl.push_back(open_kernel_cu(device,", host)
        self.assertIn('"TopPL",', host)
        self.assertIn("cu_name.c_str()", host)
        self.assertIn('open_kernel_cu(device, xrt_uuid, "Q0Ctrl", "Q0Ctrl_0")', host)
        self.assertIn("auto q0_run = q0ctrl(debug_bo, active_iters);", host)
        self.assertIn("std::vector<xrt::run> toppl_runs;", host)
        self.assertIn("toppl_runs.reserve(srad_cfg::kTopPlWorkers);", host)
        self.assertIn("toppl[worker](image_bo, output_bo, active_iters, worker)", host)
        self.assertIn("for (auto& run : toppl_runs)", host)
        self.assertIn("run.wait()", host)
        self.assertIn("q0_run.wait()", host)
        self.assertIn("graphOursPLQ0.wait()", host)
        self.assertIn("graphOursPLQ0.end()", host)
        self.assertNotIn("q0_status=", host)
        self.assertNotIn("q0_stat_phase=", host)
        self.assertNotIn("q0_send_phase=", host)

    def test_board_host_reports_split_runtime_without_renaming_pl_total(self):
        host = (
            Path(__file__).resolve().parents[1] / "ps" / "host.cpp"
        ).read_text(encoding="utf-8")

        for field in (
            "graph_init_us",
            "graph_run_submit_us",
            "q0_launch_us",
            "toppl_launch_us",
            "toppl_wait_us",
            "q0_wait_us",
            "graph_wait_us",
            "graph_end_us",
        ):
            self.assertIn(field, host)
        self.assertIn('std::printf("pl_total_us : %lld\\n", timing.wait_all_us);', host)
        self.assertIn("legacy alias of pl_total_us", host)
        self.assertIn("timing.wait_all_us = elapsed_us(pl_t0, pl_t1);", host)

    def test_board_path_uses_twelve_toppl_workers_and_q0ctrl(self):
        root = Path(__file__).resolve().parents[1]
        toppl = (root / "pl" / "TopPL.cpp").read_text(encoding="utf-8")
        q0ctrl = (root / "pl" / "Q0Ctrl.cpp").read_text(encoding="utf-8")
        host = (root / "ps" / "host.cpp").read_text(encoding="utf-8")
        conn = (root / "conn.cfg").read_text(encoding="utf-8")
        makefile = (root / "Makefile").read_text(encoding="utf-8")

        self.assertNotIn("float compute_q0sqr_from_sums", toppl)
        self.assertNotIn("const float q0sqr = compute_q0sqr_from_sums(sum, sum2);", toppl)
        self.assertIn("if ((iter & 1) == 0)", toppl)
        self.assertIn("run_worker_strip(image,", toppl)
        self.assertIn("run_worker_strip(output,", toppl)
        self.assertIn("stat_to_q0.write", toppl)
        self.assertIn("q0_from_ctrl.read", toppl)
        self.assertIn("compute_initial_worker_stats(image, active_worker, sum, sum2);", toppl)
        self.assertIn("kWorkerDataElems", toppl)
        self.assertIn("worker_col_base(worker_id)", toppl)
        self.assertIn("worker_row_base(worker_id)", toppl)
        self.assertIn("auto q0_run", host)
        self.assertIn("Q0Ctrl", conn)
        self.assertNotIn("LPDDR", conn)
        self.assertNotIn("sp=TopPL_0.image", conn)
        self.assertIn("stream_connect=TopPL_0.stat_to_q0:Q0Ctrl_0.stat_in_0:0", conn)
        self.assertIn("stream_connect=TopPL_11.stat_to_q0:Q0Ctrl_0.stat_in_11:0", conn)
        self.assertIn("stream_connect=Q0Ctrl_0.q0_out_0:TopPL_0.q0_from_ctrl:0", conn)
        self.assertIn("stream_connect=Q0Ctrl_0.q0_out_11:TopPL_11.q0_from_ctrl:0", conn)
        self.assertIn("KERNEL_XOS := $(TOPPL_XO) $(Q0CTRL_XO)", makefile)
        self.assertIn("static_assert(srad_cfg::kTopPlWorkers == 12", q0ctrl)
        self.assertIn("static_cast<float>(srad_cfg::kBoardPixels)", q0ctrl)
        self.assertIn("rm -rf _x", makefile)

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
            "kCenterValuePlane * srad_cfg::kMidPlaneStride",
            local_q,
        )
        self.assertIn(
            "kCenterValuePlane * srad_cfg::kMidPlaneStride",
            coeff_update,
        )

    def test_local_q_encodes_coefficients_and_update_decodes_them(self):
        root = Path(__file__).resolve().parents[1]
        local_q = (
            root / "aie" / "ProcessUnit" / "srad_local_q.cc"
        ).read_text(encoding="utf-8")
        coeff_update = (
            root / "aie" / "ProcessUnit" / "srad_coeff_update.cc"
        ).read_text(encoding="utf-8")

        self.assertIn("inline void encode_coeff_vec(", local_q)
        self.assertIn("aie::select(num_v, one_v, one_mask)", local_q)
        self.assertIn("*value_out = value_v.to_native();", local_q)
        self.assertIn("*tag_out = tag_v.to_native();", local_q)
        self.assertNotIn("num / den", local_q)
        self.assertNotIn("clamp01(num", local_q)
        self.assertIn("decode_coeff_vec(", coeff_update)
        self.assertIn("aie::inv(safe_den)", coeff_update)
        self.assertIn("aie::select(value_v, divided, raw_mask)", coeff_update)

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
            "store_vec(center_value, col,\n"
            "                  select_data_lanes(center_value_vec, data_mask, one_v));",
            local_q,
        )
        self.assertIn(
            "store_vec(center_tag, col,\n"
            "                  select_data_lanes(center_tag_vec, data_mask, zero_v));",
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

        self.assertIn("const float q0sqr2_scalar = q0sqr * q0sqr;", local_q)
        self.assertIn(
            "const float q0_den_scalar = q0sqr * (srad_math::kOne + q0sqr);",
            local_q,
        )
        self.assertIn("const v8float q0sqr2 = splat(q0sqr2_scalar);", local_q)
        self.assertIn("const v8float q0_den = splat(q0_den_scalar);", local_q)
        self.assertIn("const aie::vector<float, srad_cfg::kLanes> q0_den_v(q0_den);", local_q)
        self.assertIn("const auto q0_zero_mask = aie::eq(q0_den_v, zero_v);", local_q)
        self.assertIn("const auto bypass_mask = aie::eq(bypass_v, one_v);", local_q)
        self.assertIn("encode_coeff_vec(", local_q)
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
        self.assertIn("float* __restrict mid = out_c.data();", local_q)

        self.assertIn("const float* __restrict mid = in_c.data();", coeff_update)
        self.assertIn("const float* __restrict base = in_j.data();", coeff_update)
        self.assertIn("alignas(32) float out_row[srad_cfg::kOutputRowPhysElems];", coeff_update)
        self.assertIn("float* __restrict out = out_row;", coeff_update)
        self.assertIn("const uint32_t packet_id = getPacketid(out_j_next, 0);", coeff_update)
        self.assertIn("writeHeader(out_j_next, 0, packet_id);", coeff_update)
        self.assertIn("writeincr(out_j_next,", coeff_update)
        self.assertIn("chess_prepare_for_pipelining", coeff_update)
        self.assertIn("chess_loop_range", coeff_update)

    def test_coeff_update_debug_block_stays_inside_kernel_function(self):
        coeff_update = (
            Path(__file__).resolve().parents[1]
            / "aie"
            / "ProcessUnit"
            / "srad_coeff_update.cc"
        ).read_text(encoding="utf-8")

        self.assertNotIn(
            "    }\n\n#if SRAD_AIE_DEBUG && (defined(__AIESIM__) || defined(__X86SIM__))",
            coeff_update,
        )
        write_pos = coeff_update.index("writeincr(out_j_next,")
        exit_debug_pos = coeff_update.index(
            "#if SRAD_AIE_DEBUG && (defined(__AIESIM__) || defined(__X86SIM__))",
            write_pos,
        )
        self.assertGreater(exit_debug_pos, write_pos)
        self.assertTrue(coeff_update.rstrip().endswith("#endif\n}"))

    def test_local_q_uses_vector_encode_with_valid_lane_mask(self):
        root = Path(__file__).resolve().parents[1]
        local_q = (
            root / "aie" / "ProcessUnit" / "srad_local_q.cc"
        ).read_text(encoding="utf-8")

        self.assertNotIn("#include <aie_api/aie.hpp>", local_q)
        self.assertIn("inline void encode_coeff_vec(", local_q)
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
        self.assertIn("encode_coeff_vec(", local_q)
        self.assertIn("shift_left_with_zero(center_c, prev_center_c, one)", local_q)
        self.assertIn("shift_right_with_zero(center_c, center_next, one)", local_q)
        self.assertIn(
            "store_vec(south_value, col,\n"
            "                  select_data_lanes(south_value_vec, data_mask, zero_v));",
            local_q,
        )
        self.assertNotIn("fill_plane_padding(", local_q)

    def test_coeff_update_uses_vector_decode_with_valid_lane_mask(self):
        root = Path(__file__).resolve().parents[1]
        coeff_update = (
            root / "aie" / "ProcessUnit" / "srad_coeff_update.cc"
        ).read_text(encoding="utf-8")

        self.assertNotIn("#include <aie_api/aie.hpp>", coeff_update)
        self.assertIn("inline v8float decode_coeff_vec(", coeff_update)
        self.assertIn("aie::inv(safe_den)", coeff_update)
        self.assertIn("aie::select(value_v, divided, raw_mask)", coeff_update)
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
        self.assertIn("accumulate_stats(next_masked, row_sum, row_sum2);", coeff_update)
        self.assertNotIn(
            "for (int col = 1; col < srad_cfg::kRowDataElems - 1; ++col)",
            coeff_update,
        )

    def test_toppl_uses_blocking_output_reads_and_contiguous_worker_writes(self):
        toppl = (
            Path(__file__).resolve().parents[1] / "pl" / "TopPL.cpp"
        ).read_text(encoding="utf-8")

        self.assertIn("void load_16lane_input_rows(const float* current,", toppl)
        self.assertIn("void store_16lane_output_rows(float* next,", toppl)
        self.assertIn("template<int Lane>", toppl)
        self.assertIn("const plio_word_t word = from_aie_words.read();", toppl)
        self.assertNotIn("read_nb", toppl)
        self.assertNotIn("while (total_words < kLanesPerWave * kWordsPerLaneStrip)", toppl)
        self.assertNotIn("bool consume_lane_output_word(float* next,", toppl)
        self.assertRegex(
            toppl,
            r"#pragma HLS DATAFLOW disable_start_propagation\s+"
            r"load_16lane_input_rows\(current,\s+"
            r"worker_id,\s+"
            r"q0sqr,\s+"
            r"to_aie_words0,",
        )
        self.assertIn(
            "forward_input_words(to_aie_words0, out_j_0);",
            toppl,
        )
        self.assertIn(
            "capture_merged_output_packets(in_j_next_0,",
            toppl,
        )
        self.assertIn("from_aie_words0,", toppl)
        self.assertIn("from_aie_words3);", toppl)
        self.assertIn(
            "forward_input_words(to_aie_words15, out_j_15);",
            toppl,
        )
        self.assertIn(
            "capture_merged_output_packets(in_j_next_3,",
            toppl,
        )
        self.assertIn("from_aie_words12,", toppl)
        self.assertIn("from_aie_words15);", toppl)
        self.assertIn(
            "store_16lane_output_rows(next, worker_id,",
            toppl,
        )
        self.assertIn("for (int stream_row = 0;", toppl)
        self.assertIn("to_aie_words0.write(pack_two_floats(", toppl)
        self.assertIn("to_aie_words15.write(pack_two_floats(", toppl)
        self.assertIn("next[row_base + elem] = row_elems[elem];", toppl)
        self.assertNotIn("store_one_lane_output_word(", toppl)
        self.assertNotIn("void make_16lane_output_requests(", toppl)
        self.assertNotIn(
            "run_one_lane_strip(current, next, out_j_0, in_j_next_0,",
            toppl,
        )

    def test_toppl_keeps_lane_output_addressing_static_without_polling_state(self):
        toppl = (
            Path(__file__).resolve().parents[1] / "pl" / "TopPL.cpp"
        ).read_text(encoding="utf-8")

        self.assertNotIn("template<int LaneBase>", toppl)
        self.assertNotIn("lane_stream_to_board_row_const", toppl)
        self.assertNotIn("lane_output_valid_const", toppl)
        self.assertIn("template<int Lane>", toppl)
        self.assertNotIn("strip_for_lane", toppl)
        self.assertIn("worker_col_base(worker_id)", toppl)
        self.assertIn("const int first_row", toppl)
        self.assertNotIn("struct LaneOutputState", toppl)
        self.assertNotIn("advance_lane_output_state", toppl)
        self.assertIn("const int physical_col = word_col * 2", toppl)
        self.assertNotIn("word_index / kWordsPerRow", toppl)
        self.assertNotIn("word_value_at(", toppl)
        self.assertNotIn("right_half", toppl)
        self.assertIn("const int row_base =", toppl)
        self.assertIn("next[row_base + elem] = row_elems[elem];", toppl)
        self.assertIn(
            "physical_col == srad_cfg::kStatSumPadIndex - 1",
            toppl,
        )
        self.assertIn(
            "physical_col == srad_cfg::kStatSum2PadIndex",
            toppl,
        )

    def test_toppl_feeds_input_lanes_round_robin_to_match_output_merge(self):
        toppl = (
            Path(__file__).resolve().parents[1] / "pl" / "TopPL.cpp"
        ).read_text(encoding="utf-8")

        self.assertNotIn("plio_word_t load_one_lane_input_word(const float* current,", toppl)
        self.assertIn("void load_16lane_input_rows(", toppl)
        self.assertNotIn("void load_left_half_input_rows(", toppl)
        self.assertNotIn("void load_right_half_input_rows(", toppl)
        self.assertNotIn("strip_for_lane", toppl)
        self.assertIn("const int col_base = worker_col_base(worker_id);", toppl)
        self.assertIn("for (int stream_row = 0;", toppl)
        self.assertIn("for (int word_col = 0;", toppl)
        self.assertLess(
            toppl.index("to_aie_words0.write(pack_two_floats("),
            toppl.index("to_aie_words1.write(pack_two_floats("),
        )
        self.assertLess(
            toppl.index("to_aie_words1.write(pack_two_floats("),
            toppl.index("to_aie_words15.write(pack_two_floats("),
        )
        self.assertIn("load_16lane_input_rows(current,", toppl)
        self.assertNotIn("store_one_lane_output_rows(next,", toppl)

    def test_toppl_runs_one_sixteen_lane_worker_half_at_a_time(self):
        toppl = (
            Path(__file__).resolve().parents[1] / "pl" / "TopPL.cpp"
        ).read_text(encoding="utf-8")

        self.assertIn("constexpr int kLanesPerWorker = srad_cfg::kLanesPerTopPl", toppl)
        self.assertIn("constexpr int kWorkerDataElems =", toppl)
        self.assertNotIn("constexpr int kStripBatches", toppl)
        self.assertNotIn("int strip_for_lane", toppl)
        self.assertIn("void load_16lane_input_rows(const float* current,", toppl)
        self.assertIn("void store_16lane_output_rows(float* next,", toppl)
        self.assertIn("void run_one_strip_batch_16lanes(", toppl)
        self.assertNotIn("for (int strip_batch = 0;", toppl)
        self.assertNotIn("strip_batch < kStripBatches", toppl)
        self.assertIn("run_one_strip_batch_16lanes(current,", toppl)
        self.assertIn("out_j[0],", toppl)
        self.assertIn("out_j[15],", toppl)
        self.assertIn("in_j_next[3],", toppl)
        self.assertNotIn("in_j_next[4],", toppl)
        self.assertNotIn("run_one_strip_4lane_wave", toppl)
        self.assertNotIn("lane_row_begin", toppl)
        self.assertNotIn("lane_row_end", toppl)
        self.assertNotIn("run_one_lane_strip<11>", toppl)

    def test_toppl_avoids_nested_lane_dataflow_control_for_timing(self):
        toppl = (
            Path(__file__).resolve().parents[1] / "pl" / "TopPL.cpp"
        ).read_text(encoding="utf-8")

        self.assertIn("#pragma HLS DATAFLOW disable_start_propagation", toppl)
        self.assertNotIn("void run_one_lane_stream(", toppl)
        self.assertNotIn("run_one_lane_stream(to_aie_words", toppl)
        self.assertIn("forward_input_words(to_aie_words0, out_j_0);", toppl)
        self.assertIn(
            "capture_merged_output_packets(in_j_next_0,",
            toppl,
        )

    def test_toppl_limits_m_axi_adapter_buffering_for_timing(self):
        toppl = (
            Path(__file__).resolve().parents[1] / "pl" / "TopPL.cpp"
        ).read_text(encoding="utf-8")
        normalized = toppl.replace("\\\n", " ")

        for port, bundle in (
            ("image", "gmem0"),
            ("output", "gmem1"),
        ):
            self.assertRegex(
                normalized,
                rf"#pragma HLS INTERFACE m_axi port={port} offset=slave bundle={bundle}\s+"
                r"max_read_burst_length=16 max_write_burst_length=16\s+"
                r"num_read_outstanding=1 num_write_outstanding=1\s+"
                r"max_widen_bitwidth=128",
            )

    def test_toppl_uses_resource_light_dataflow_fifos_for_192lane_impl(self):
        toppl = (
            Path(__file__).resolve().parents[1] / "pl" / "TopPL.cpp"
        ).read_text(encoding="utf-8")

        self.assertIn("#pragma HLS STREAM variable=to_aie_words0 depth=16", toppl)
        self.assertIn("#pragma HLS STREAM variable=from_aie_words15 depth=64", toppl)
        self.assertIn(
            "#pragma HLS bind_storage variable=to_aie_words0 type=fifo impl=srl",
            toppl,
        )
        self.assertIn(
            "#pragma HLS bind_storage variable=from_aie_words15 type=fifo impl=lutram",
            toppl,
        )
        self.assertNotIn("type=fifo impl=bram", toppl)

    def test_classic_vpp_kernel_compile_does_not_use_hls_component_cfg(self):
        makefile = (
            Path(__file__).resolve().parents[1] / "Makefile"
        ).read_text(encoding="utf-8")

        self.assertNotIn("--config $(TOPPL_CFG)", makefile)
        self.assertNotIn("--config $(Q0CTRL_CFG)", makefile)
        self.assertNotIn("TOPPL_CFG :=", makefile)
        self.assertNotIn("Q0CTRL_CFG :=", makefile)
        self.assertIn(
            "$(VPP) $(VPP_XO_FLAGS) -k $(TOPPL_KERNEL) -o $(TOPPL_XO) $(TOPPL_SRC)",
            makefile,
        )
        self.assertIn(
            "$(VPP) $(VPP_XO_FLAGS) -k $(Q0CTRL_KERNEL) -o $(Q0CTRL_XO) $(Q0CTRL_SRC)",
            makefile,
        )

    def test_toppl_avoids_hls_unsupported_m_axi_pointer_selection(self):
        toppl = (
            Path(__file__).resolve().parents[1] / "pl" / "TopPL.cpp"
        ).read_text(encoding="utf-8")

        self.assertNotIn("const float* current = (iter & 1) ? output : image;", toppl)
        self.assertNotIn("float* next = (iter & 1) ? image : output;", toppl)
        self.assertNotIn("const float* current_l = (iter & 1) ? output_l : image_l;", toppl)
        self.assertNotIn("float* next_l = (iter & 1) ? image_l : output_l;", toppl)
        self.assertIn("run_worker_strip(image,", toppl)
        self.assertIn("run_worker_strip(output,", toppl)

    def test_board_config_uses_192_lanes_twelve_workers_and_six_row_blocks(self):
        cfg, _lambda_default, _bypass = verify_srad.read_config()

        self.assertEqual(cfg["kRows"], 125)
        self.assertEqual(cfg["kCols"], 125)
        self.assertEqual(cfg["kRowPhysElems"], 128)
        self.assertEqual(cfg["kBoardRows"], 4000)
        self.assertEqual(cfg["kBoardCols"], 4000)
        self.assertEqual(cfg["kBoardIterations"], 100)
        self.assertEqual(cfg["kBoardStrips"], 32)
        self.assertEqual(cfg["kTopPlWorkers"], 12)
        self.assertEqual(cfg["kTopPlColumnWorkers"], 2)
        self.assertEqual(cfg["kBoardRowBlocks"], 6)
        self.assertEqual(cfg["kRowsPerRowBlock"], 667)
        self.assertEqual(cfg["kBoardPaddedRows"], 4002)
        self.assertEqual(cfg["kLanesPerTopPl"], 16)
        self.assertEqual(cfg["kParallelLanes"], 192)
        self.assertEqual(cfg["kAieLaneCols"], 48)
        self.assertEqual(cfg["kOutputMergeWays"], 4)
        self.assertEqual(cfg["kMergedOutputPlioCount"], 48)
        self.assertEqual(cfg["kMergedOutputsPerTopPl"], 4)
        self.assertEqual(cfg["kBoardStripsPerTopPl"], 16)
        self.assertEqual(cfg["kWorkerCols"], 2000)
        self.assertEqual(cfg["kBoardStripBatches"], 1)
        self.assertEqual(cfg["kBoardRowsPerLaneMax"], 667)
        self.assertEqual(cfg["kBoardRowsPerStrip"], 670)
        self.assertEqual(
            cfg["kBoardGraphRowsPerIteration"],
            cfg["kBoardRowsPerStrip"],
        )

    def test_conn_uses_four_way_output_merge_and_congestion_impl_directives(self):
        root = Path(__file__).resolve().parents[1]
        conn = (root / "conn.cfg").read_text(encoding="utf-8")
        toppl = (root / "pl" / "TopPL.cpp").read_text(encoding="utf-8")

        self.assertIn(
            "stream_connect=ai_engine_0.ours_plq0_out_j_next_merged_0:TopPL_0.in_j_next_0:0",
            conn,
        )
        self.assertIn(
            "stream_connect=ai_engine_0.ours_plq0_out_j_next_merged_47:TopPL_11.in_j_next_3:0",
            conn,
        )
        self.assertNotRegex(conn, r"^stream_connect=.*:(16|32)$", re.MULTILINE)
        self.assertNotIn("ours_plq0_out_j_next_merged_48", conn)
        self.assertIn("return static_cast<int>(header.range(1, 0));", toppl)
        self.assertIn(
            "prop=run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=AltSpreadLogic_high",
            conn,
        )
        self.assertIn(
            "prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=AggressiveFanoutOpt",
            conn,
        )
        self.assertIn(
            "prop=run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=AggressiveExplore",
            conn,
        )
        self.assertIn(
            "prop=run.impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED=1",
            conn,
        )
        self.assertIn(
            "prop=run.impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE=AggressiveExplore",
            conn,
        )

    def test_toppl_row_blocks_have_internal_halo_context(self):
        cfg, _lambda_default, _bypass = verify_srad.read_config()

        self.assertEqual(cfg["kBoardLanePreContextRows"], 1)
        self.assertEqual(cfg["kFlushRows"], 2)
        self.assertEqual(
            cfg["kBoardRowsPerLaneStream"],
            cfg["kBoardLanePreContextRows"]
            + cfg["kRowsPerRowBlock"]
            + cfg["kFlushRows"],
        )

        blocks = []
        for row_block in range(cfg["kBoardRowBlocks"]):
            center_first = row_block * cfg["kRowsPerRowBlock"]
            center_last = center_first + cfg["kRowsPerRowBlock"] - 1
            input_first = center_first - cfg["kBoardLanePreContextRows"]
            input_last = center_last + cfg["kFlushRows"]
            blocks.append((center_first, center_last, input_first, input_last))

        self.assertEqual(blocks[0], (0, 666, -1, 668))
        self.assertEqual(blocks[1], (667, 1333, 666, 1335))
        self.assertEqual(blocks[5], (3335, 4001, 3334, 4003))

    def test_aie_fifo_depths_allow_k1_k2_alignment_slack(self):
        cfg, _lambda_default, _bypass = verify_srad.read_config()

        self.assertGreaterEqual(cfg["kInputObjectFifoDepth"], 4)
        self.assertGreaterEqual(cfg["kDelayedInputObjectFifoDepth"], 4)
        self.assertGreaterEqual(cfg["kMidObjectFifoDepth"], 4)
        self.assertGreaterEqual(cfg["kOutputObjectFifoDepth"], 4)

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
        self.assertIn('f"plio_ours_j_{lane}.txt"', gen_case_text)
        self.assertIn('default=base / "input_image_sim.txt"', verify_text)
        self.assertIn('default=base / "aiesim_j_next_merged_0.txt"', verify_text)
        self.assertIn("python $(DATA_GEN)", makefile)
        self.assertIn("python $(DATA_GEN) --sim", makefile)
        merged_ids = self._makefile_variable(makefile, "MERGED_OUTPUT_IDS")
        self.assertIn("44 45 46 47", merged_ids)
        self.assertNotIn("48", merged_ids.split())
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

        self.assertIn("SIM_OUTPUT := $(foreach lane", makefile)
        merged_ids = self._makefile_variable(makefile, "MERGED_OUTPUT_IDS")
        self.assertIn("44 45 46 47", merged_ids)
        self.assertNotIn("48", merged_ids.split())
        self.assertIn("./data/aiesim_j_next_merged_$(lane).txt", makefile)
        self.assertIn("SIM_OUTPUT_DIRS :=", makefile)
        self.assertIn("rm -f $(SIM_OUTPUT)", makefile)
        self.assertIn("rm -rf $(SIM_OUTPUT_DIRS)", makefile)

    def test_package_depends_on_linked_xsa_and_graph_archive(self):
        makefile = (
            Path(__file__).resolve().parents[1] / "Makefile"
        ).read_text(encoding="utf-8")
        normalized = makefile.replace("\\\n", " ")

        self.assertRegex(
            normalized,
            r"\$\(XCLBIN\):[^\n]*\$\(XSA\)[^\n]*\$\(GRAPH_O\)",
        )

    def test_xsa_depends_on_fresh_xo_and_graph_outputs(self):
        makefile = (
            Path(__file__).resolve().parents[1] / "Makefile"
        ).read_text(encoding="utf-8")
        normalized = makefile.replace("\\\n", " ")

        self.assertRegex(
            normalized,
            r"\$\(TOPPL_XO\):[^\n]*\$\(TOPPL_SRC\)[^\n]*\./aie/Config\.h",
        )
        self.assertRegex(
            normalized,
            r"\$\(Q0CTRL_XO\):[^\n]*\$\(Q0CTRL_SRC\)[^\n]*\./aie/Config\.h",
        )
        self.assertRegex(
            normalized,
            r"\$\(XSA\):[^\n]*\$\(GRAPH_O\)[^\n]*\$\(KERNEL_XOS\)",
        )

    def _makefile_variable(self, makefile: str, name: str) -> str:
        match = re.search(rf"^{name}\s*:=\s*(.*?)(?=^\S|\Z)", makefile, re.M | re.S)
        self.assertIsNotNone(match, f"{name} not found")
        return match.group(1).replace("\\\n", " ")

    def test_verify_decodes_four_way_packet_merged_output(self):
        cfg, _lambda_default, _bypass = verify_srad.read_config()
        rows_per_iter = cfg["kRowsPerIterSim"]
        row_phys = cfg["kRowPhysElems"]
        merge_ways = cfg["kOutputMergeWays"]

        def float_bits(value: float) -> int:
            return struct.unpack("<I", struct.pack("<f", value))[0]

        expected = [
            [
                float(way * 100000 + i)
                for i in range(rows_per_iter * row_phys)
            ]
            for way in range(merge_ways)
        ]

        words = []
        for row in range(rows_per_iter):
            for way in (2, 0, 3, 1):
                words.append(way)
                row_base = row * row_phys
                for value in expected[way][row_base:row_base + row_phys]:
                    words.append(float_bits(value))

        decoded = verify_srad.decode_merged_pktstream_words(words, cfg)

        self.assertEqual(decoded, expected)

    def test_verify_resolves_x86simulator_output_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requested = root / "data" / "aiesim_j_next_0.txt"
            produced = root / "x86simulator_output" / "data" / "aiesim_j_next_0.txt"
            produced.parent.mkdir(parents=True)
            produced.write_text("1.0 2.0\n", encoding="utf-8")

            self.assertEqual(
                verify_srad.resolve_aie_output_path(requested),
                produced,
            )

    def test_verify_resolves_work_x86simulator_output_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requested = root / "data" / "aiesim_j_next_0.txt"
            produced = (
                root
                / "Work"
                / "x86simulator_output"
                / "data"
                / "aiesim_j_next_0.txt"
            )
            produced.parent.mkdir(parents=True)
            produced.write_text("1.0 2.0\n", encoding="utf-8")

            self.assertEqual(
                verify_srad.resolve_aie_output_path(requested),
                produced,
            )

    def test_verify_reports_checked_output_locations_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            requested = Path(tmp) / "data" / "aiesim_j_next_0.txt"

            with self.assertRaises(FileNotFoundError) as ctx:
                verify_srad.resolve_aie_output_path(requested)

            message = str(ctx.exception)
            self.assertIn("AIE output not found", message)
            self.assertIn("x86simulator_output", message)


if __name__ == "__main__":
    unittest.main()
