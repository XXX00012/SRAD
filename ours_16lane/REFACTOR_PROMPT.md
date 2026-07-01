# ours 行流式 SRAD 改造提示词（给 codex）

## 0. 你的任务与角色

你要把 `SRAD/ours` 这个 VCK190 工程从**“tile 式 + 全 halo”**改造成**“行流式 + margin”**，对齐 `stencilflow/tri_1plio` 的数据流理念。目标是标定“单 AIE lane 处理一行 256 → 输出一行 256 需要多少时间”，为后续 4000×4000×100 大图设计 toppl/lane 数量提供吞吐数据。

工程根目录：`d:/科研/AIE代码/SRAD/ours`。参考工程：`d:/科研/AIE代码/stencilflow/tri_1plio`（margin 机制、PLIO 流式 PL、host 计时的范本）。

**重要约束：**
- 本地只有源代码，没有编译输出。你**不要**尝试运行 `make`／`v++`／`aiesimulator`，这些都在远程跑。你只改源码，改完清晰说明改了哪些文件、为什么。
- 改之前先读：`aie/Config.h`、`aie/ProcessUnit/{srad.h,include.h,srad_local_q.cc,srad_coeff_update.cc}`、`aie/ProcessGraph/StencilCoreGraph.h`、`aie/TopGraph.{h,cpp}`、`pl/{TopPL.cpp,Q0Ctrl.cpp,TopPL.cfg}`、`conn.cfg`、`ps/host.cpp`、`data/gen_case.py`、`Makefile`，以及参考工程 `stencilflow/tri_1plio` 的对应文件。
- 对照 AMD 手册：`D:/科研/AMD指南/UG1079.pdf`、`D:/科研/AMD指南/UG1399.pdf`（优先检索 `D:/科研/AMD指南/UG1399.txt`）确认 `input_buffer` + `margin<>` 的语义、PLIO 位宽规则、`adf::event` profiling API。
- 数据类型保持 **float32**，PLIO 保持 **plio_64_bits**（每 word 打包 2 个 float）。

## 1. 设计决策（已和用户敲定，不要改动）

1. **AIE 输入单元**：从 19×24 物理 tile 改成**一行 258 个 float**：前 256 是真实数据，第 257 个（索引 256）放 q0sqr，第 258 个（索引 257）补 0 对齐 64bit。输出仍是一行纯 256 float（输出不带 q0）。
2. **邻居来源**：不再由 PL 打 halo。改用 ADF `input_buffer<float, extents<inherited_extent>, margin<N>>`，由 AIE 自动保留前 N 行历史做 N/S 邻居。
3. **margin 行数（两核不同）**：
   - **K1 `srad_local_q` margin = 3 行**（要算 center(r)/south=coeff(r+1)/east(r)，并集需 J 的 r-1,r,r+1,r+2 共 4 行，喂入行为最南行则留前 3 行）。
   - **K2 `srad_coeff_update` margin = 2 行**（重算 dN/dS/dW/dE 需 J 的 r-1,r,r+1 共 3 行，留前 2 行）。
4. **两核分工（选①，沿用旧 ours 已验证协议的行流式翻版）**：
   - K1 每行算 center/south/east 三组 value-tag 系数包，输出系数流。
   - K2 收 K1 系数包 + 原始 J 行，解码系数后做 divergence 更新，输出 J_next 一行。
   - **不要**改成 K2 自己重算 coeff 的端到端缓存方案。
5. **q0 路径（方案 A）**：保留 Q0Ctrl 全图统计反馈环。每轮迭代 TopPL 先逐行扫当前图算 sum/sum2，Q0Ctrl 归约出全图 q0sqr，下一轮 TopPL 把 q0 塞进每行第 257 位喂 AIE。**不要**用 s_axilite/RTP 传 q0（那只适用于 q0 是编译期/启动前常量的情况，和反馈环冲突）。
6. **多轮迭代**：保留现有 N 轮 SRAD ping-pong（image↔output 交替）。
7. **边界策略**：margin 区域越界**填 0**，不做 clamp。边界几行结果错了无所谓。
8. **make all（上板）**：256×256，host 跑全量。**make sim（仿真）**：复用 256×256 大数据，但只跑前 20 行（256×20），仅跑通不验正确性。

## 2. 数据布局（精确规格）

### 2.1 一行输入（喂给 AIE 的 PLIO 流，每行）

```
索引:  0   1   2  ... 255 | 256 | 257
内容:  d0  d1  d2 ... d255 | q0  | 0(pad)
       <--- 256 真实数据 --> <q0> <对齐补0>
共 258 个 float = 129 个 PLIO 64bit word（258/2=129，偶数，对齐 OK）
```

- 当前行 = 第 r 行的 256 个 J 数据。
- 第 256 位（第 257 个数）= 本轮全图 q0sqr，每行都带同一个值。
- 第 257 位（第 258 个数）= 0，仅为 64bit 对齐填充。

### 2.2 一行输出（AIE → PL → DDR，每行）

```
纯 256 个 float = 128 个 PLIO 64bit word。输出不带 q0、不带 stat。
```

> 注意：旧 ours 的输出 tile 末尾附带 tile_sum/tile_sum2 两个统计值。行流式下，**stat 的产生位置要重新设计**——见 §5.3，统计由 PL 在“扫描当前图”阶段算，不再由 AIE kernel 在输出流里附带。如果你判断让 AIE 顺带算每行 sum/sum2 更简单，可以保留并在每行输出后附 2 个 stat（那样一行输出 258），但要在提示里和用户确认前**默认采用 PL 扫描方案**，保持 AIE 输出纯净 256。

### 2.3 margin 在内存里的语义

ADF `input_buffer<float, extents<inherited_extent>, margin<M>>`，其中 `M` 是 margin 的**元素数** = margin行数 × 258。`in.data()` 指向缓冲区起始，前 M 个元素是上几次 firing 的历史行（margin），紧接着是本次 firing 的当前行 258 个元素。参考 `tri_1plio/aie/ProcessUnit/hdiff.h` 的 `lap_input_buffer` 写法（它的 margin = `kWindowMarginRows * kRowElems`）。

- K1：`kK1MarginElems = 3 * 258`。窗口可见 4 行：`row[-3], row[-2], row[-1], row[0]=当前行`。
- K2：`kK2MarginElems = 2 * 258`。窗口可见 3 行：`row[-2], row[-1], row[0]=当前行`。

## 3. Config.h 重写要点

`aie/Config.h` 现在充满 tile/halo 相关常量和 static_assert。删掉 tile 体系，建立行流式体系。新增/保留的关键常量（命名建议，可微调但要全工程一致）：

```cpp
namespace srad_cfg {

// ---- 图像尺寸 ----
constexpr int kRows = 256;          // make all 上板尺寸
constexpr int kCols = 256;
constexpr int kPixels = kRows * kCols;

// ---- 仿真行数 ----
constexpr int kSimRows = 20;        // make sim 只跑前 20 行

// ---- 行布局 ----
constexpr int kDataCols = 256;            // 每行真实数据
constexpr int kQ0Slot = 256;              // q0 放在索引 256
constexpr int kRowElems = 258;            // 256 数据 + q0 + pad，对齐 64bit
constexpr int kOutputRowElems = 256;      // 输出纯 256
constexpr int kLanes = 8;                 // v8float 向量宽度

// ---- margin ----
constexpr int kK1MarginRows = 3;
constexpr int kK2MarginRows = 2;
constexpr int kK1MarginElems = kK1MarginRows * kRowElems;  // 3*258
constexpr int kK2MarginElems = kK2MarginRows * kRowElems;  // 2*258

// ---- 系数包（K1→K2）----
// 沿用旧 ours 的 center/south/east 各 value+tag，共 6 个 plane，但现在按“一行”组织。
constexpr int kMidRecordElems = 6;
constexpr int kMidElemsPerRow = kOutputRowElems * kMidRecordElems; // 256*6

// ---- 并行/多轮 ----
constexpr int kParallelLanes = 4;     // 维持 4 lane（按现有 conn.cfg/host）
constexpr int kTopPlWorkers = 4;
constexpr int kSradIterations = 3;    // 多轮迭代轮数，保留
constexpr float kLambdaDefault = 0.5f;

// ---- 数据类型 ----
constexpr int kScalarBytes = sizeof(float);

}
```

**要点：**
- 删除所有 `kHalo*`、`kOutputTile*`、`kInputLogical*`、`kInputPhysical*`、`kQ0SqrTileCol/Row/Index`、`kTile*`、`kGraphRunIterations`（tile 版）等 tile 专属常量及其 static_assert。
- 保留 `kParallelLanes/kTopPlWorkers/kSradIterations/kLambdaDefault`，多轮迭代和 4-lane 结构沿用。
- 新增 static_assert：`kRowElems % 2 == 0`（64bit PLIO）、`kRowElems == kDataCols + 2`、`kDataCols % kLanes == 0`、`kOutputRowElems % kLanes == 0`、`kK1MarginRows > kK2MarginRows`。
- `kRows/kCols` 是上板尺寸，4000×4000 时只改这两个值；行布局常量不变。

## 4. srad.h / include.h 改造

### 4.1 srad.h（buffer 类型加 margin）

```cpp
#pragma once
#include <adf.h>
#include "Config.h"
using namespace adf;

// K1 的 J 输入：margin 3 行
using srad_j_k1_buffer = input_buffer<
    float, adf::extents<adf::inherited_extent>,
    adf::margin<srad_cfg::kK1MarginElems>>;

// K2 的 J 输入：margin 2 行
using srad_j_k2_buffer = input_buffer<
    float, adf::extents<adf::inherited_extent>,
    adf::margin<srad_cfg::kK2MarginElems>>;

// K1→K2 系数包：无 margin（按行传递）
using srad_mid_input_buffer = input_buffer<float, adf::extents<adf::inherited_extent>>;

extern "C" {
void srad_local_q(srad_j_k1_buffer& in_j, output_buffer<float>& out_c);
void srad_coeff_update(srad_mid_input_buffer& in_c,
                       srad_j_k2_buffer& in_j,
                       output_buffer<float>& out_j_next);
}
```

参考 `tri_1plio/aie/ProcessUnit/hdiff.h` 的 margin buffer 写法确认语法。

### 4.2 include.h（宏改行流式）

把 `ROW`/`COL`/`TILE_SIZE` 等宏从 tile 尺寸改成行尺寸：
- `COL` → `srad_cfg::kRowElems`（258，kernel 内部按行索引用）
- `OUT_COL` → `srad_cfg::kOutputRowElems`（256）
- 删掉 halo 相关宏。保留 `srad_math` 命名空间里的常量（kZero/kOne/kQuarter/kHalf/kOneSixteenth）。
- `srad_math` 里的 `north_row/south_row/west_col/east_col/image_index` 这些按全图坐标的 helper，行流式 kernel 里用不到，可删或保留供 host 参考用（host 仍需全图 CPU 参考）。

## 5. kernel 改造（核心，最关键）

旧 kernel 是“一次 firing 处理整个 16×16 tile，内部 16 行循环”。新 kernel 是“一次 firing 处理**一行** 256，靠 margin 拿到上几行”。每次 graph firing 输入一行、输出一行。

### 5.1 K1 `srad_local_q.cc`（每行算 center/south/east 系数包）

输入：`in_j.data()` 指向 `[3 行 margin][当前行]`，每行 258 float。可见 4 行，记为：
```
rm3 = data + 0*258   // row[-3]
rm2 = data + 1*258   // row[-2]
rm1 = data + 2*258   // row[-1]
r0  = data + 3*258   // 当前行 row[0]
```
q0sqr 从当前行取：`q0sqr = r0[256]`（kQ0Slot）。

**算法语义（保持与旧 ours / gen_case.py 的 compute_c 完全一致）：**
- 旧 ours 一次 firing 为 tile 内每个 (out_r) 同时算三组系数：`center=coeff(r)`、`south=coeff(r+1)`、`east=coeff(r,c+1)`。行流式下，**让“当前 firing 的输出行”对应被 K2 更新的那一行**。设当前要服务的更新行是 `R`，则：
  - `center 行` = coeff(R)：需 J 的 R-1, R, R+1
  - `south 行` = coeff(R+1)：需 J 的 R, R+1, R+2
  - `east` = center 行内左移一位（同行邻居）
- 因此 K1 在一次 firing 里，用可见的 4 行窗口算出 **针对更新行 R 的 center/south/east**。**关键：确定 margin 窗口里哪一行是 R。** 由于 margin=3、可见 r-3..r0，且 south 需要 R+2，令 **R = row[-1]（即 rm1）**，则 R-1=rm2, R+1=r0... 不对，这样 R+2 超出窗口。

  **正确映射**（务必按此实现）：当前 firing 喂入的“当前行 r0”是全局第 `t` 行。我们让 K1 这次 firing 产出的系数服务于更新行 `R = t - 2`：
  - center(R)=coeff(t-2)：需 J(t-3)=rm3, J(t-2)=rm2, J(t-1)=rm1 ✅ 全在窗口
  - south(R)=coeff(t-1)：需 J(t-2)=rm2, J(t-1)=rm1, J(t)=r0 ✅ 全在窗口
  - east(R)：center 行(t-2)内左右邻 ✅
  - 这正是 margin=3（窗口 4 行 rm3..r0）的来由。**K1 输出的系数包对应更新行 R=t-2，存在 2 行延迟（warmup）。**

- **越界填 0**：当某历史行对应的全局行号 < 0（开头几次 firing），margin 区域是未初始化/0。要保证语义上越界邻居取 0。具体做法：让 PL 在流的最前面**预喂 3 行 0**（warmup 行），使第一次有效更新行从全局 row 0 开始正确对齐；或在 kernel 内对 firing 序号 < warmup 的输出不做要求（反正边界不验正确性）。**采用 PL 预喂 warmup 行的方式**（见 §6），kernel 不需要特判。

**向量化**：256 列 = 32 个 v8float chunk。沿用旧 `srad_local_q.cc` 的 `compute_coeff_vec_at<C>` / `encode_coeff_vec` / fpmac 写法，但：
- 行缓存 `CachedJRow` 的长度从 `kInputRowElems`(24) 改成能放下 258 或至少 256+左右 halo 的长度。**注意行内 E/W 邻居**：c=0 的 west 和 c=255 的 east 越界，按填 0 处理（旧代码用 west_col_local/east_col_local 做 clamp，这里改成越界返回 0 或在行两端补 0 槽）。
- 系数包输出布局：`out_c.data()` 一行 = 6 个 plane × 256 = `kMidElemsPerRow`。plane 顺序沿用旧：center_value, center_tag, south_value, south_tag, east_value, east_tag（各 256）。

### 5.2 K2 `srad_coeff_update.cc`（每行解码系数 + 更新）

输入：
- `in_c.data()`：K1 传来的一行系数包（6×256）。
- `in_j.data()`：margin 2 行 + 当前行，可见 3 行：`rm2, rm1, r0`（每行 258）。
- 让 K2 这次 firing 更新的行 `R = t' - 1`（t' 是 K2 当前喂入行号），则 center 行=R 需 J(R-1)=rm2... 实际：K2 重算 dN/dS/dW/dE 用 `jn=J(R-1), jc=J(R), js=J(R+1)`。令 **R = rm1（row[-1]）**：jn=rm2, jc=rm1, js=r0 ✅，margin=2 窗口 3 行正好。
- **K1 和 K2 的延迟对齐**：K1 系数包对应更新行 R=t-2，K2 也要在更新同一个 R 时消费它。graph 的 fifo 和 run 次数要保证两核对同一 R 对齐。沿用旧 ours 用 fifo_depth + 延迟输入对齐的思路（旧代码有 `kDelayedInputObjectFifoDepth`、`kUpdateRowLagRows`）。**这是最容易错的地方，务必在 graph 里把 K1→K2 系数流和 K2 的 J 输入流的相对延迟对齐到同一更新行。**

更新公式（与 gen_case.py srad_next / host compute 完全一致）：
```
dN=jn-jc; dS=js-jc; dW=jw-jc; dE=je-jc   // jw/je 同行左右，越界填0
D = coeff*dN + coeff_south*dS + coeff*dW + coeff_east*dE
out = jc + 0.25 * lambda * D
```
其中 coeff/coeff_south/coeff_east 来自 K1 系数包解码（沿用旧 `decode_coeff_vec`：tag>0 时 value/tag，否则 value 原样）。

输出：一行纯 256 float 写 `out_j_next`。

### 5.3 stat（sum/sum2）的产生

旧 ours 由 K2 在输出 tile 末尾附 tile_sum/tile_sum2，PL 汇总给 Q0Ctrl。行流式下**默认改为：PL 在每轮迭代开始前，单独逐行扫描当前图算 sum/sum2**（见 §6），AIE 输出保持纯 256。这样 AIE kernel 不掺统计逻辑，最干净。

> 若 codex 评估后认为“AIE 顺带每行算 sum/sum2 并附在输出后（一行输出 258）”实现更简单且不影响计时，可在改造说明里**提出来让用户定夺**，但默认实现 PL 扫描方案。

## 6. PL 改造：TopPL.cpp（大改成逐行）

删掉所有 tile 逻辑（`tile_origin`、`write_one_tile`、`read_j_next_tile`、`maybe_store_valid`、`tile_linear_*`、`copy_*_tile` 等）。改成 tri_1plio 式的逐行 DDR↔PLIO 流，但**保留**多轮迭代外层循环、Q0Ctrl 反馈握手、4-worker 分工。

参考 `tri_1plio/pl/TopPL.cpp` 的 `read_ddr_words / pack_to_plio / unpack_from_plio / write_ddr_words` + `#pragma HLS DATAFLOW` 结构。

### 6.1 每行打包（write 方向，DDR → AIE）

```
对当前 worker 负责的每一行 r（含 warmup 预喂行）:
  读 DDR 该行 256 个 float
  打包 129 个 64bit word:
    word[0..127]: (d0,d1),(d2,d3)...(d254,d255)   // 256 数据 → 128 word
    word[128]:    (q0sqr, 0.0f)                    // q0 + pad
  写入 out_j 流
```
- q0sqr 来自上一轮 Q0Ctrl 反馈（`q0_from_ctrl.read()`），每轮一个值，该轮所有行复用。
- **warmup 预喂**：每个 lane 的行流最前面预喂 `kK1MarginRows=3` 行（内容全 0，q0 照填），让 margin 窗口在第一有效行就对齐。输出端对应丢弃前若干 warmup 行（边界不验正确性，丢弃逻辑简单处理即可）。

### 6.2 每行回收（read 方向，AIE → DDR）

```
对每一有效输出行:
  从 in_j_next 流读 128 个 word
  解包成 256 个 float，写回 DDR 对应行
```

### 6.3 行带分 lane（多 lane 切分）

256×256 全图按行带切给 4 个 worker：worker w 负责 `行 r where r % kParallelLanes == w` 或连续行带（`w*rows/4 .. (w+1)*rows/4`）。**采用连续行带**更利于 margin 历史连续（每个 worker 处理自己行带，行带交界处的 margin 用邻带行或填 0——边界不验正确性，填 0 即可）。在改造说明里写清你选了哪种切法。

### 6.4 stat 扫描（每轮迭代前）

保留 `compute_initial_worker_stats` 类似逻辑，但按行扫描当前图的 256 数据算 sum/sum2，`stat_to_q0.write(pack_two_floats(sum,sum2))`。多轮迭代里每轮更新后重新扫描。

### 6.5 多轮 ping-pong

保留旧 TopPL 的 `image↔output` 交替（iter 偶/奇切换源缓冲）、`q0_from_ctrl.read()` 每轮握手、`stat_to_q0.write` 每轮上报的结构。把内部的 tile 循环换成行循环即可。

### 6.6 TopPL 接口签名

保持与旧版一致的端口（image/output AXI master、out_j/in_j_next/stat_to_q0/q0_from_ctrl 四条 AXIS、iter_cnt/worker_id_arg 的 s_axilite），这样 `conn.cfg` 和 host 基本不动。仅内部实现改逐行。

## 7. Q0Ctrl.cpp

基本保留。它从 4 个 worker 收 partial sum/sum2、归约、算 q0sqr、广播回 4 个 worker。`compute_q0sqr_from_sums` 里的 `kPixels` 在 256×256 下仍是全图像素数，逻辑不变。确认 `kPixels` 引用的是新 Config 的 `kRows*kCols`。

## 8. graph：StencilCoreGraph.h / TopGraph.{h,cpp}

### 8.1 StencilCoreGraph.h

- 维持 4 lane、每 lane 两个 kernel（k_local_q→k_coeff_update）的结构。
- `dimensions` 改成行尺寸：
  - `k_local_q.in[0]` = `{kRowElems}`（258，margin 在 buffer 类型里声明，dimensions 给的是每次 firing 的 sample 大小）
  - `k_local_q.out[0]` = `{kMidElemsPerRow}`（6×256）
  - `k_coeff_update.in[0]` = `{kMidElemsPerRow}`
  - `k_coeff_update.in[1]` = `{kRowElems}`（258）
  - `k_coeff_update.out[0]` = `{kOutputRowElems}`（256）
- **延迟对齐**：用 `fifo_depth` 把 K1→K2 系数流和 K2 的 J 输入流对齐到同一更新行 R（见 §5.2）。参考旧 ours 的 `kDelayedInputObjectFifoDepth` 思路，给 K2 的 J 输入流足够 depth 以吸收 K1 的 2 行算系数延迟。在注释里写清延迟行数。
- 确认 margin buffer 与 dimensions 的配合：margin 由 buffer 类型 `margin<N>` 提供，`dimensions` 只声明 per-firing sample。参照 tri_1plio 的 StencilCoreGraph.h（它的 lap 输入 dimensions 是 `kLapInputSampleElems=kRowElems`，margin 在 hdiff.h 的 buffer 类型里）。

### 8.2 TopGraph.h/.cpp

- PLIO 保持 `plio_64_bits`。
- `graph.run(N)` 的 N = 每 lane 要处理的行数 ×（warmup 调整）× 迭代轮数。具体：单轮每 lane 行数 = `kRows/kParallelLanes + kK1MarginRows`（含 warmup），多轮 ×`kSradIterations`。在注释里写清 run 次数怎么算出来的。
- aiesim 的 main（`#if defined(__AIESIM__)...`）：`graph.run(kSimRows 对应的行数)`，只跑前 20 行验证流程跑通。dump 逻辑相应改成按行 dump。

## 9. host.cpp（ps/host.cpp）

参考 `tri_1plio/ps/host.cpp`（用户已在 IDE 选中过，那是干净范本）：
- 计时用 `adf::event::start_profiling(out_plio[0], io_stream_start_to_bytes_transferred_cycles, output_bytes)` + `read_profiling`，并输出 `aie_out_cycles_row`（每行周期）、`aie_out_us_est`（按 1.25GHz 估算）。这是标定“一行吞吐”的核心指标，**务必保留并打印 per-row 数据**。
- `iter_cnt` 语义：仍是 SRAD 迭代轮数（1..kSradIterations），不要和“行数”混淆。行数由 Config 的 kRows 决定。
- 保留 4 worker + Q0Ctrl 的提交/等待结构（旧 host 的多线程 wait 逻辑可保留）。
- CPU 参考 `cpu_reference_iterations`：256×256 全量算太慢可只在 make sim 小规模时启用；上板 256×256 可保留但注意耗时。**make all 上板时正确性不是重点**，可以把 compare 输出留着但不作为成败判据。
- DDR 布局：input/output BO 现在是行主序 `kRows*kCols` 个 float（256×256），不再有 tile 重排。BO 大小相应调整。
- 打印里所有 “tile/halo/19x24” 字样改成行流式描述（256 数据+q0+pad / margin K1=3 K2=2 / 每行 258→256）。

## 10. data/gen_case.py

- 删掉 `make_j_tile_stream` / `make_one_tile_stream` / tile 相关读取。
- 生成 256×256 行主序数据文件：每行 256 个 float（host 读进 DDR）。仍用原 `image[r][c] = 1.0 + 0.003r + 0.002c + 0.05*sin(0.31r)*cos(0.19c)` 公式。
- 保留 `compute_q0sqr` / `srad_next` / `compute_c`（CPU 参考/golden 生成）。
- 如果 aiesim 需要 PLIO 输入文件（258/行格式），生成对应的 `input_plio` 文件：每行 129 个 64bit-pair 行（256 数据成对 + (q0,0)）。参照 tri 的 input_plio.txt 格式（每行两个数对应一个 64bit word）。
- 从 Config.h 读维度的逻辑（`read_config_dims`）要更新 required 常量列表，去掉已删除的 tile 常量，加入 kRows/kCols/kRowElems/kDataCols/kK1MarginRows/kK2MarginRows/kSradIterations 等。

## 11. Makefile / conn.cfg

### 11.1 Makefile
- `PLIO_DATA_FILES` 改成新的行流式数据文件名（去掉 tile 文件）。
- `make all` = aie kernels xsa host package sd_card（不变），上板 256×256。
- `make sim`：`AIESIM` 跑前 20 行。可在 sim 目标或 Config 里用 kSimRows 控制 graph.run 次数。确认 sim 用的数据文件就是 all 的大文件（前 20 行被读）。
- 频率 `AIE_FREQ=450MHz` 维持。

### 11.2 conn.cfg
- 端口名若没改就**不动**（TopPL 接口签名保持，见 §6.6）。
- 若 TopPL/Q0Ctrl 的 AXIS 端口名有任何变化，同步更新 stream_connect。
- 维持 4 TopPL CU + 1 Q0Ctrl + 4 AIE lane 的连接拓扑。

## 12. 涉及文件清单（确认改了哪些 + 为什么）

改完后，请按下表逐项报告你改了什么、为什么改、有没有偏离本提示词：

| 文件 | 预期改动 |
|---|---|
| `aie/Config.h` | 重写：删 tile/halo，建行流式常量（kRows/kCols=256, kRowElems=258, margin, kSimRows） |
| `aie/ProcessUnit/srad.h` | buffer 加 margin（K1=3行/K2=2行） |
| `aie/ProcessUnit/include.h` | ROW/COL 宏改行尺寸，删 halo 宏 |
| `aie/ProcessUnit/srad_local_q.cc` | 改逐行算 center/south/east 系数包，更新行 R=t-2 映射 |
| `aie/ProcessUnit/srad_coeff_update.cc` | 改逐行解码系数+更新，更新行 R=t'-1 映射 |
| `aie/ProcessGraph/StencilCoreGraph.h` | dimensions 改行尺寸，fifo 延迟对齐 |
| `aie/TopGraph.h` / `aie/TopGraph.cpp` | run 次数按行算，aiesim 跑 20 行，dump 按行 |
| `pl/TopPL.cpp` | 大改逐行 DDR↔PLIO，保留多轮+反馈+4worker |
| `pl/Q0Ctrl.cpp` | 基本保留，确认 kPixels 引用对 |
| `ps/host.cpp` | per-row 计时，DDR 行主序，打印改行流式 |
| `data/gen_case.py` | 生成 256×256 行主序 + PLIO 258/行，删 tile |
| `Makefile` | 数据文件名、sim 跑 20 行 |
| `conn.cfg` | 端口不变则不动 |

## 13. 验证与交付

- 你**不要**跑 make。改完后：
  1. 自检每个文件改动是否前后一致（常量名、维度、margin 数值全工程对得上）。
  2. 重点自查 §5.2 的**两核延迟对齐**：K1 系数包对应的更新行 R 和 K2 消费时的 R 必须是同一行，否则结果全错。把你的对齐推导写出来。
  3. 重点自查 **margin 元素数**：K1=3×258、K2=2×258，buffer 类型、Config、graph dimensions 三处一致。
  4. 列出所有你**不确定**或**做了取舍**的点（尤其 stat 产生方式、行带切分、warmup 行数、延迟对齐），交给用户定夺。
- 交付：改动文件清单 + 每个文件的关键 diff 说明 + 不确定点列表。

## 14. 红线（不要做）

- 不要把 float 改成 int32。
- 不要用 s_axilite/RTP 传 q0（和反馈环冲突）。
- 不要删掉 Q0Ctrl 反馈环或多轮迭代。
- 不要改成 K2 端到端缓存方案（用户选了 K1 传系数包的选①）。
- 不要跑编译/仿真命令。
- 不确定的算法语义，对照 `gen_case.py` 的 `compute_c`/`srad_next` 和 `host.cpp` 的 `cpu_reference`——它们是 SRAD 数值语义的权威参考。
