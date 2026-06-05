#include <adf.h>
#include <aie_api/aie.hpp>

#include "ProcessUnit/include.h"
#include "ProcessUnit/srad.h"

using namespace adf;

namespace {

constexpr unsigned kLaneOffsets = 0x76543210;
constexpr int kChunksPerRow = BLOCK_COL / srad_cfg::kLanes;
constexpr int kManualSpillRowsPerGroup = 4;

#define SRAD_NOINLINE __attribute__((noinline))

static_assert(kChunksPerRow == 2,
              "8-vector pressure kernel assumes 16-column tiles");
static_assert(kManualSpillRowsPerGroup == 4,
              "pressure kernel keeps four rows live per update group");
static_assert(BLOCK_ROW == 16,
              "fused pressure kernel assumes one 16-row tile group");

inline v8float load8(const float* __restrict ptr) {
    return *(reinterpret_cast<const v8float*>(ptr));
}

inline void store8(float* __restrict ptr, v8float v) {
    *(reinterpret_cast<v8float*>(ptr)) = v;
}

inline v8float load_spill8(const volatile float* __restrict ptr) {
    return *(reinterpret_cast<const volatile v8float*>(ptr));
}

inline void store_spill8(volatile float* __restrict ptr, v8float v) {
    *(reinterpret_cast<volatile v8float*>(ptr)) = v;
}

inline volatile float* spill_slot(volatile float* base, int slot) {
    return base + slot * srad_cfg::kLanes;
}

inline const volatile float* spill_slot(const volatile float* base, int slot) {
    return base + slot * srad_cfg::kLanes;
}

inline void spill_store_slot(volatile float* base, int slot, v8float value) {
    store_spill8(spill_slot(base, slot), value);
}

inline v8float spill_load_slot(const volatile float* base, int slot) {
    return load_spill8(spill_slot(base, slot));
}

SRAD_NOINLINE v8float splat8(float value) {
    alignas(32) float tmp[8] = {
        value, value, value, value, value, value, value, value};
    return load8(tmp);
}

inline v8float vadd(v8float acc, v8float x, v8float ones) {
    return fpmac(acc, x, 0, kLaneOffsets, ones, 0, kLaneOffsets);
}

inline v8float vsub(v8float a, v8float b, v8float neg_one) {
    return fpmac(a, b, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
}

inline v8float vmul(v8float a, v8float b) {
    return fpmul(a, 0, kLaneOffsets, b, 0, kLaneOffsets);
}

inline int block_index(int r, int c) {
    return r * BLOCK_COL + c;
}

enum SpillDir {
    kDirN = 0,
    kDirS = 1,
    kDirW = 2,
    kDirE = 3,
};

constexpr int kSpillGradSlotBase = 0;
constexpr int kSpillGradRows = kManualSpillRowsPerGroup;
constexpr int kSpillDirsPerChunk = 4;
constexpr int kSpillGradSlots =
    kSpillGradRows * kChunksPerRow * kSpillDirsPerChunk;
constexpr int kSpillCoeffSlotBase = kSpillGradSlotBase + kSpillGradSlots;
constexpr int kSpillCoeffRows = kManualSpillRowsPerGroup + 1;
constexpr int kSpillCoeffSlots = kSpillCoeffRows * kChunksPerRow;
constexpr int kSpillDeltaSlotBase = kSpillCoeffSlotBase + kSpillCoeffSlots;
constexpr int kSpillDeltaSlots = kSpillGradRows * kChunksPerRow;
constexpr int kSpillWorkSlotBase = kSpillDeltaSlotBase + kSpillDeltaSlots;
constexpr int kSpillWorkSlots = 26;
constexpr int kManualSpillSlots = kSpillWorkSlotBase + kSpillWorkSlots;

constexpr int kWorkDn = 0;
constexpr int kWorkDs = 1;
constexpr int kWorkDw = 2;
constexpr int kWorkDe = 3;
constexpr int kWorkCe = 4;
constexpr int kWorkNum = 5;
constexpr int kWorkDen = 6;
constexpr int kWorkTmpBase = 8;

static_assert(kSpillGradRows == kManualSpillRowsPerGroup,
              "manual spill layout assumes four update rows per group");
static_assert(kSpillCoeffRows == kManualSpillRowsPerGroup + 1,
              "manual spill layout needs the south coefficient row");
static_assert(kWorkDen < kWorkTmpBase,
              "manual spill named work slots must not overlap temporaries");
static_assert(kWorkTmpBase + 18 <= kSpillWorkSlots,
              "manual spill work area is too small for num/den temporaries");

inline int grad_slot(int local_row, int chunk, int dir) {
    return kSpillGradSlotBase +
           ((local_row * kChunksPerRow + chunk) * kSpillDirsPerChunk + dir);
}

inline int coeff_slot(int coeff_row, int chunk) {
    return kSpillCoeffSlotBase + coeff_row * kChunksPerRow + chunk;
}

inline int delta_slot(int local_row, int chunk) {
    return kSpillDeltaSlotBase + local_row * kChunksPerRow + chunk;
}

inline int work_slot(int rel) {
    return kSpillWorkSlotBase + rel;
}

SRAD_NOINLINE void build_lr8(const float* __restrict row,
                             int col_base,
                             float* __restrict west,
                             float* __restrict east) {
    if (col_base == 0) {
        west[0] = row[0];
        west[1] = row[0];
        west[2] = row[1];
        west[3] = row[2];
        west[4] = row[3];
        west[5] = row[4];
        west[6] = row[5];
        west[7] = row[6];

        east[0] = row[1];
        east[1] = row[2];
        east[2] = row[3];
        east[3] = row[4];
        east[4] = row[5];
        east[5] = row[6];
        east[6] = row[7];
        east[7] = row[8];
    } else {
        west[0] = row[7];
        west[1] = row[8];
        west[2] = row[9];
        west[3] = row[10];
        west[4] = row[11];
        west[5] = row[12];
        west[6] = row[13];
        west[7] = row[14];

        east[0] = row[9];
        east[1] = row[10];
        east[2] = row[11];
        east[3] = row[12];
        east[4] = row[13];
        east[5] = row[14];
        east[6] = row[15];
        east[7] = row[15];
    }
}

SRAD_NOINLINE void clamp_compute_c_spilled(
    const volatile float* __restrict num_c,
    const volatile float* __restrict den_c,
    float* __restrict div_out,
    volatile float* __restrict spill,
    int coeff_out_slot) {
    for (int lane = 0; lane < srad_cfg::kLanes; ++lane) {
        const float num = num_c[lane];
        const float den = den_c[lane];
        if (den <= srad_math::kZero) {
            div_out[lane] = srad_math::kZero;
        } else if (den < num) {
            div_out[lane] = srad_math::kOne;
        } else {
            div_out[lane] = num / den;
        }
    }

    spill_store_slot(spill, coeff_out_slot, load8(div_out));
}

SRAD_NOINLINE void compute_num_den8_spilled(v8float jc,
                                            const volatile float* d_n,
                                            const volatile float* d_s,
                                            const volatile float* d_w,
                                            const volatile float* d_e,
                                            volatile float* spill,
                                            int num_out_slot,
                                            int den_out_slot,
                                            v8float q0sqr2,
                                            v8float q0_den,
                                            v8float one,
                                            v8float neg_one,
                                            v8float quarter,
                                            v8float half,
                                            v8float one_sixteenth) {
    const int t = kWorkTmpBase;

    spill_store_slot(spill, work_slot(t + 0),
                     vmul(load_spill8(d_n), load_spill8(d_n)));
    spill_store_slot(spill, work_slot(t + 1),
                     vmul(load_spill8(d_s), load_spill8(d_s)));
    spill_store_slot(spill, work_slot(t + 2),
                     vadd(spill_load_slot(spill, work_slot(t + 0)),
                          spill_load_slot(spill, work_slot(t + 1)), one));

    spill_store_slot(spill, work_slot(t + 3),
                     vmul(load_spill8(d_w), load_spill8(d_w)));
    spill_store_slot(spill, work_slot(t + 4),
                     vmul(load_spill8(d_e), load_spill8(d_e)));
    spill_store_slot(spill, work_slot(t + 5),
                     vadd(spill_load_slot(spill, work_slot(t + 3)),
                          spill_load_slot(spill, work_slot(t + 4)), one));
    spill_store_slot(spill, work_slot(t + 6),
                     vadd(spill_load_slot(spill, work_slot(t + 2)),
                          spill_load_slot(spill, work_slot(t + 5)), one));

    spill_store_slot(spill, work_slot(t + 7),
                     vadd(load_spill8(d_n), load_spill8(d_s), one));
    spill_store_slot(spill, work_slot(t + 8),
                     vadd(load_spill8(d_w), load_spill8(d_e), one));
    spill_store_slot(spill, work_slot(t + 9),
                     vadd(spill_load_slot(spill, work_slot(t + 7)),
                          spill_load_slot(spill, work_slot(t + 8)), one));

    spill_store_slot(spill, work_slot(t + 10),
                     vmul(half, spill_load_slot(spill, work_slot(t + 6))));
    spill_store_slot(spill, work_slot(t + 11),
                     vmul(spill_load_slot(spill, work_slot(t + 9)),
                          spill_load_slot(spill, work_slot(t + 9))));
    spill_store_slot(spill, work_slot(t + 12),
                     vmul(one_sixteenth,
                          spill_load_slot(spill, work_slot(t + 11))));
    spill_store_slot(spill, work_slot(t + 13),
                     vsub(spill_load_slot(spill, work_slot(t + 10)),
                          spill_load_slot(spill, work_slot(t + 12)),
                          neg_one));

    spill_store_slot(spill, work_slot(t + 14),
                     vmul(quarter, spill_load_slot(spill, work_slot(t + 9))));
    spill_store_slot(spill, work_slot(t + 15),
                     vadd(jc, spill_load_slot(spill, work_slot(t + 14)),
                          one));
    spill_store_slot(spill, work_slot(t + 16),
                     vmul(spill_load_slot(spill, work_slot(t + 15)),
                          spill_load_slot(spill, work_slot(t + 15))));
    spill_store_slot(spill, work_slot(t + 17),
                     vmul(q0sqr2, spill_load_slot(spill, work_slot(t + 16))));
    spill_store_slot(spill, den_out_slot,
                     vadd(spill_load_slot(spill, work_slot(t + 13)),
                          spill_load_slot(spill, work_slot(t + 17)), one));
    spill_store_slot(spill, num_out_slot,
                     vmul(q0_den, spill_load_slot(spill, work_slot(t + 16))));
}

SRAD_NOINLINE void compute_update8_spilled(const volatile float* c,
                                           const volatile float* cs,
                                           const volatile float* ce,
                                           const volatile float* d_n,
                                           const volatile float* d_s,
                                           const volatile float* d_w,
                                           const volatile float* d_e,
                                           volatile float* spill,
                                           int delta_out_slot,
                                           v8float one) {
    const int t = kWorkTmpBase;

    spill_store_slot(spill, work_slot(t + 0),
                     vmul(load_spill8(c), load_spill8(d_n)));
    spill_store_slot(spill, work_slot(t + 1),
                     vmul(load_spill8(cs), load_spill8(d_s)));
    spill_store_slot(spill, work_slot(t + 2),
                     vadd(spill_load_slot(spill, work_slot(t + 0)),
                          spill_load_slot(spill, work_slot(t + 1)), one));
    spill_store_slot(spill, work_slot(t + 3),
                     vmul(load_spill8(c), load_spill8(d_w)));
    spill_store_slot(spill, work_slot(t + 4),
                     vmul(load_spill8(ce), load_spill8(d_e)));
    spill_store_slot(spill, work_slot(t + 5),
                     vadd(spill_load_slot(spill, work_slot(t + 3)),
                          spill_load_slot(spill, work_slot(t + 4)), one));
    spill_store_slot(spill, delta_out_slot,
                     vadd(spill_load_slot(spill, work_slot(t + 2)),
                          spill_load_slot(spill, work_slot(t + 5)), one));
}

SRAD_NOINLINE void update_j8_spilled(v8float jc,
                                     const volatile float* d,
                                     volatile float* spill,
                                     float* __restrict out,
                                     v8float update_scale,
                                     v8float one) {
    const int t = kWorkTmpBase;
    spill_store_slot(spill, work_slot(t + 0),
                     vmul(update_scale, load_spill8(d)));
    store8(out, vadd(jc, spill_load_slot(spill, work_slot(t + 0)), one));
}

SRAD_NOINLINE void compute_coeff_chunk_spilled(v8float jc,
                                               v8float north,
                                               v8float south,
                                               int col_base,
                                               int chunk,
                                               float* __restrict row_j,
                                               float* __restrict row_w,
                                               float* __restrict row_e,
                                               float* __restrict div_out,
                                               volatile float* spill,
                                               int coeff_row,
                                               v8float q0sqr2,
                                               v8float q0_den,
                                               v8float one,
                                               v8float neg_one,
                                               v8float quarter,
                                               v8float half,
                                               v8float one_sixteenth) {
    build_lr8(row_j, col_base, row_w, row_e);
    spill_store_slot(spill, work_slot(kWorkDn),
                     vsub(north, jc, neg_one));
    spill_store_slot(spill, work_slot(kWorkDs),
                     vsub(south, jc, neg_one));
    spill_store_slot(spill, work_slot(kWorkDw),
                     vsub(load8(row_w), jc, neg_one));
    spill_store_slot(spill, work_slot(kWorkDe),
                     vsub(load8(row_e), jc, neg_one));

    if constexpr (srad_cfg::kBypassCoeffMath) {
        spill_store_slot(spill, coeff_slot(coeff_row, chunk), one);
    } else {
        compute_num_den8_spilled(
            jc, spill_slot(spill, work_slot(kWorkDn)),
            spill_slot(spill, work_slot(kWorkDs)),
            spill_slot(spill, work_slot(kWorkDw)),
            spill_slot(spill, work_slot(kWorkDe)), spill,
            work_slot(kWorkNum), work_slot(kWorkDen), q0sqr2, q0_den, one,
            neg_one, quarter, half, one_sixteenth);
        clamp_compute_c_spilled(
            spill_slot(spill, work_slot(kWorkNum)),
            spill_slot(spill, work_slot(kWorkDen)), div_out, spill,
            coeff_slot(coeff_row, chunk));
    }
}

SRAD_NOINLINE void compute_coeff_row_pair_spilled(v8float jc0,
                                                  v8float jc1,
                                                  v8float north0,
                                                  v8float north1,
                                                  v8float south0,
                                                  v8float south1,
                                                  float* __restrict row_j,
                                                  float* __restrict row_w,
                                                  float* __restrict row_e,
                                                  float* __restrict div_out,
                                                  volatile float* spill,
                                                  int coeff_row,
                                                  v8float q0sqr2,
                                                  v8float q0_den,
                                                  v8float one,
                                                  v8float neg_one,
                                                  v8float quarter,
                                                  v8float half,
                                                  v8float one_sixteenth) {
    store8(row_j, jc0);
    store8(row_j + 8, jc1);

    compute_coeff_chunk_spilled(jc0, north0, south0, 0, 0, row_j, row_w,
                                row_e, div_out, spill, coeff_row, q0sqr2,
                                q0_den, one, neg_one, quarter, half,
                                one_sixteenth);
    compute_coeff_chunk_spilled(jc1, north1, south1, 8, 1, row_j, row_w,
                                row_e, div_out, spill, coeff_row, q0sqr2,
                                q0_den, one, neg_one, quarter, half,
                                one_sixteenth);
}

SRAD_NOINLINE void compute_update_grad_chunk_spilled(v8float jc,
                                                     v8float north,
                                                     v8float south,
                                                     int col_base,
                                                     int chunk,
                                                     float* __restrict row_j,
                                                     float* __restrict row_w,
                                                     float* __restrict row_e,
                                                     volatile float* spill,
                                                     int grad_row,
                                                     v8float neg_one) {
    build_lr8(row_j, col_base, row_w, row_e);
    spill_store_slot(spill, grad_slot(grad_row, chunk, kDirN),
                     vsub(north, jc, neg_one));
    spill_store_slot(spill, grad_slot(grad_row, chunk, kDirS),
                     vsub(south, jc, neg_one));
    spill_store_slot(spill, grad_slot(grad_row, chunk, kDirW),
                     vsub(load8(row_w), jc, neg_one));
    spill_store_slot(spill, grad_slot(grad_row, chunk, kDirE),
                     vsub(load8(row_e), jc, neg_one));
}

SRAD_NOINLINE void compute_update_grad_row_pair_spilled(v8float jc0,
                                                        v8float jc1,
                                                        v8float north0,
                                                        v8float north1,
                                                        v8float south0,
                                                        v8float south1,
                                                        float* __restrict row_j,
                                                        float* __restrict row_w,
                                                        float* __restrict row_e,
                                                        volatile float* spill,
                                                        int grad_row,
                                                        v8float neg_one) {
    store8(row_j, jc0);
    store8(row_j + 8, jc1);

    compute_update_grad_chunk_spilled(jc0, north0, south0, 0, 0, row_j,
                                      row_w, row_e, spill, grad_row, neg_one);
    compute_update_grad_chunk_spilled(jc1, north1, south1, 8, 1, row_j,
                                      row_w, row_e, spill, grad_row, neg_one);
}

SRAD_NOINLINE void compute_output_chunk_spilled(v8float jc,
                                                int col_base,
                                                int chunk,
                                                float* __restrict out,
                                                float* __restrict c_row,
                                                float* __restrict row_ce,
                                                float* __restrict cw_tmp,
                                                volatile float* spill,
                                                int grad_row,
                                                int coeff_row,
                                                int south_coeff_row,
                                                v8float update_scale,
                                                v8float one) {
    build_lr8(c_row, col_base, cw_tmp, row_ce);
    spill_store_slot(spill, work_slot(kWorkCe), load8(row_ce));
    compute_update8_spilled(
        spill_slot(spill, coeff_slot(coeff_row, chunk)),
        spill_slot(spill, coeff_slot(south_coeff_row, chunk)),
        spill_slot(spill, work_slot(kWorkCe)),
        spill_slot(spill, grad_slot(grad_row, chunk, kDirN)),
        spill_slot(spill, grad_slot(grad_row, chunk, kDirS)),
        spill_slot(spill, grad_slot(grad_row, chunk, kDirW)),
        spill_slot(spill, grad_slot(grad_row, chunk, kDirE)), spill,
        delta_slot(grad_row, chunk), one);
    update_j8_spilled(jc, spill_slot(spill, delta_slot(grad_row, chunk)),
                      spill, out, update_scale, one);
}

SRAD_NOINLINE void compute_output_row_pair_spilled(v8float jc0,
                                                   v8float jc1,
                                                   float* __restrict out0,
                                                   float* __restrict out1,
                                                   float* __restrict c_row,
                                                   float* __restrict row_ce,
                                                   float* __restrict cw_tmp,
                                                   volatile float* spill,
                                                   int grad_row,
                                                   int coeff_row,
                                                   int south_coeff_row,
                                                   v8float update_scale,
                                                   v8float one) {
    store8(c_row, spill_load_slot(spill, coeff_slot(coeff_row, 0)));
    store8(c_row + 8, spill_load_slot(spill, coeff_slot(coeff_row, 1)));

    compute_output_chunk_spilled(jc0, 0, 0, out0, c_row, row_ce, cw_tmp,
                                 spill, grad_row, coeff_row,
                                 south_coeff_row, update_scale, one);
    compute_output_chunk_spilled(jc1, 8, 1, out1, c_row, row_ce, cw_tmp,
                                 spill, grad_row, coeff_row,
                                 south_coeff_row, update_scale, one);
}

#define SRAD_LOAD_ROW_REL(ID, ROW_EXPR)                                      \
    const int row_##ID = (ROW_EXPR);                                         \
    const v8float j_##ID##_c0 = load8(j_base + block_index(row_##ID, 0));    \
    const v8float j_##ID##_c1 = load8(j_base + block_index(row_##ID, 8));    \
    const v8float u_##ID##_c0 =                                              \
        load8(j_update_base + block_index(row_##ID, 0));                     \
    const v8float u_##ID##_c1 =                                              \
        load8(j_update_base + block_index(row_##ID, 8))

#define SRAD_LOAD_J_ROW_REL(ID, ROW_EXPR)                                    \
    const int row_##ID = (ROW_EXPR);                                         \
    const v8float j_##ID##_c0 = load8(j_base + block_index(row_##ID, 0));    \
    const v8float j_##ID##_c1 = load8(j_base + block_index(row_##ID, 8))

#define SRAD_COMPUTE_DU_REL(ID, LOCAL_ROW, N, S)                             \
    compute_update_grad_row_pair_spilled(                                    \
        u_##ID##_c0, u_##ID##_c1, u_##N##_c0, u_##N##_c1,                  \
        u_##S##_c0, u_##S##_c1, row_j, row_w, row_e, spill, LOCAL_ROW,      \
        neg_one)

#define SRAD_COMPUTE_C_REL(ID, COEFF_ROW, N, S)                              \
    compute_coeff_row_pair_spilled(                                          \
        j_##ID##_c0, j_##ID##_c1, j_##N##_c0, j_##N##_c1,                  \
        j_##S##_c0, j_##S##_c1, row_j, row_w, row_e, div_out, spill,        \
        COEFF_ROW, q0sqr2, q0_den, one, neg_one, quarter, half,             \
        one_sixteenth)

#define SRAD_COMPUTE_OUT_REL(ID, LOCAL_ROW, COEFF_ROW, SOUTH_COEFF_ROW)      \
    compute_output_row_pair_spilled(                                         \
        u_##ID##_c0, u_##ID##_c1, out_base + block_index(row_##ID, 0),      \
        out_base + block_index(row_##ID, 8), c_row, row_ce, row_w, spill,   \
        LOCAL_ROW, COEFF_ROW, SOUTH_COEFF_ROW, update_scale, one)

SRAD_NOINLINE void process_4row_spill_pressure_block(
    const float* __restrict j_base,
    const float* __restrict j_update_base,
    float* __restrict row_j,
    float* __restrict row_w,
    float* __restrict row_e,
    float* __restrict c_row,
    float* __restrict row_ce,
    float* __restrict div_out,
    volatile float* spill,
    float* __restrict out_base,
    v8float q0sqr2,
    v8float q0_den,
    v8float one,
    v8float neg_one,
    v8float quarter,
    v8float half,
    v8float one_sixteenth,
    v8float update_scale,
    int start_row) {
    SRAD_LOAD_ROW_REL(prev, (start_row == 0) ? 0 : start_row - 1);
    SRAD_LOAD_ROW_REL(r0, start_row + 0);
    SRAD_LOAD_ROW_REL(r1, start_row + 1);
    SRAD_LOAD_ROW_REL(r2, start_row + 2);
    SRAD_LOAD_ROW_REL(r3, start_row + 3);
    SRAD_LOAD_ROW_REL(next, (start_row + 4 >= BLOCK_ROW) ?
                            BLOCK_ROW - 1 : start_row + 4);

    SRAD_COMPUTE_DU_REL(r0, 0, prev, r1);
    SRAD_COMPUTE_DU_REL(r1, 1, r0, r2);
    SRAD_COMPUTE_DU_REL(r2, 2, r1, r3);
    SRAD_COMPUTE_DU_REL(r3, 3, r2, next);

    SRAD_COMPUTE_C_REL(r0, 0, prev, r1);
    SRAD_COMPUTE_C_REL(r1, 1, r0, r2);
    SRAD_COMPUTE_C_REL(r2, 2, r1, r3);
    SRAD_COMPUTE_C_REL(r3, 3, r2, next);
    if (start_row + 4 >= BLOCK_ROW) {
        SRAD_COMPUTE_C_REL(next, 4, r2, next);
    } else {
        SRAD_LOAD_J_ROW_REL(next_south, start_row + 5);
        SRAD_COMPUTE_C_REL(next, 4, r3, next_south);
    }

    SRAD_COMPUTE_OUT_REL(r0, 0, 0, 1);
    SRAD_COMPUTE_OUT_REL(r1, 1, 1, 2);
    SRAD_COMPUTE_OUT_REL(r2, 2, 2, 3);
    SRAD_COMPUTE_OUT_REL(r3, 3, 3, 4);
}

#undef SRAD_LOAD_ROW_REL
#undef SRAD_LOAD_J_ROW_REL
#undef SRAD_COMPUTE_DU_REL
#undef SRAD_COMPUTE_C_REL
#undef SRAD_COMPUTE_OUT_REL

} // namespace

#undef SRAD_NOINLINE

void srad_undivide_fused(srad_image_input_buffer& in_j,
                         srad_image_input_buffer& in_j_update,
                         srad_scalar_input_buffer& in_q0sqr,
                         srad_scalar_input_buffer& in_lambda,
                         output_buffer<float>& out_j_next) {
    const float* __restrict j_base = in_j.data();
    const float* __restrict j_update_base = in_j_update.data();
    const float* __restrict q0_base = in_q0sqr.data();
    const float* __restrict lambda_base = in_lambda.data();
    float* __restrict out_base = out_j_next.data();

    const float q0sqr = q0_base[0];
    const float lambda = lambda_base[0];

    const v8float q0 = splat8(q0sqr);
    const v8float q0sqr2 = vmul(q0, q0);
    const v8float one = splat8(srad_math::kOne);
    const v8float neg_one = splat8(-srad_math::kOne);
    const v8float quarter = splat8(srad_math::kQuarter);
    const v8float half = splat8(srad_math::kHalf);
    const v8float one_sixteenth = splat8(srad_math::kOneSixteenth);
    const v8float q0_den = vmul(q0, vadd(one, q0, one));
    const v8float update_scale = splat8(srad_math::kQuarter * lambda);

    alignas(32) float row_j[BLOCK_COL];
    alignas(32) float row_w[8], row_e[8];
    alignas(32) float div_out[8];
    alignas(32) float c_row[BLOCK_COL];
    alignas(32) float row_ce[8];
    alignas(32) volatile float spill[kManualSpillSlots * srad_cfg::kLanes];

    for (int start_row = 0; start_row < BLOCK_ROW;
         start_row += kManualSpillRowsPerGroup) {
        process_4row_spill_pressure_block(
            j_base, j_update_base, row_j, row_w, row_e, c_row, row_ce,
            div_out, spill, out_base, q0sqr2, q0_den, one, neg_one, quarter,
            half, one_sixteenth, update_scale, start_row);
    }
}
