# TT-Metal Kernel Generator - Debug Prompt

**Generated at:** 2025-11-11 15:21:10
**Kernel Type:** reader
**Operation:** diode_equation
**Core Mode:** single
**Model:** llama-3.3-70b-versatile

## System Prompt (RAG Context)

```
You are an expert TT-Metal kernel developer for Tenstorrent's Wormhole architecture.

# TT-METAL COMPUTE API

## eltwise_binary.h:
namespace ckernel {
PACK((llk_pack_init(ocb)));
ALWI void binary_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false) {
if constexpr (full_init) {
ALWI void mul_tiles_init(uint32_t icb0, uint32_t icb1) { binary_tiles_init<true, ELWMUL>(icb0, icb1); }
ALWI void add_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false) {
ALWI void sub_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false) {
ALWI void mul_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
ALWI void add_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
ALWI void sub_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
ALWI void binary_dest_reuse_tiles_init(uint32_t icb0) {
ALWI void binary_dest_reuse_tiles(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
}  // namespace ckernel

## eltwise_binary_sfpu.h:
namespace ckernel {
ALWI void add_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
ALWI void sub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
ALWI void mul_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
ALWI void div_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
ALWI void rsub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
ALWI void power_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
ALWI void add_binary_tile_init() { MATH((llk_math_eltwise_binary_sfpu_binop_init<APPROX, ckernel::BinaryOp::ADD>())); }
ALWI void sub_binary_tile_init() { MATH((llk_math_eltwise_binary_sfpu_binop_init<APPROX, ckernel::BinaryOp::SUB>())); }
ALWI void mul_binary_tile_init() { MATH((llk_math_eltwise_binary_sfpu_binop_init<APPROX, ckernel::BinaryOp::MUL>())); }
ALWI void div_binary_tile_init() { MATH((llk_math_eltwise_binary_sfpu_binop_init<APPROX, ckernel::BinaryOp::DIV>())); }
ALWI void rsub_binary_tile_init() {
ALWI void power_binary_tile_init() {
}  // namespace ckernel

## eltwise_unary.h:
namespace ckernel {
}  // namespace ckernel

## exp.h:
namespace ckernel {
ALWI void exp_tile_init() {
ALWI void exp_tile(uint32_t idst, int vector_mode = (int)VectorMode::RC, uint16_t scale = p_sfpu::kCONST_1_FP16B) {
}  // namespace ckernel

## cb_api.h:
namespace ckernel {
* of tiles. Important note: in case multiple calls of cb_wait_front(n) are issued without a paired cb_pop_front() call,
* cb_wait_front(8) followed by a cb_pop_front(32) would produce incorrect behavior. Instead 4 calls of cb_wait_front()
* all cb_wait_front calls in the same kernel. Example 1: cb_wait_front(32), cb_wait_front(40), cb_pop_front(32+8) tiles
* on a CB of size 64 would produce incorrect behavior. Example 2: cb_wait_front(3) on a cb of size 32 would also
ALWI void cb_wait_front(uint32_t cbid, uint32_t ntiles) { UNPACK((llk_wait_tiles(cbid, ntiles))); }
* can only be updated from one thread at a time. Example: if compute kernel has cb_pop_front(input_id, 1)
* and writer kernel also has cb_pop_front(input_id, 1), these calls will produce non-deterministic behavior because
ALWI void cb_pop_front(uint32_t cbid, uint32_t ntiles) { UNPACK((llk_pop_tiles(cbid, ntiles))); }
ALWI void cb_reserve_back(uint32_t cbid, uint32_t ntiles) {
* can only be updated from one thread at a time. Example: if compute kernel has cb_push_back(output_id, 1)
* and reader kernel also has cb_push_back(output_id, 1), these calls will produce non-deterministic behavior because
ALWI void cb_push_back(uint32_t cbid, uint32_t ntiles) { PACK((llk_push_tiles<false, false>(cbid, ntiles))); }
ALWI void cb_get_tile(uint32_t cb_id, uint32_t index, volatile void* p_tile) {
UNPACK(llk_unpack_get_tile(cb_id, index, (uint32_t*)p_tile));
MATH(llk_math_get_tile(cb_id, index, (uint32_t*)p_tile));
PACK(llk_pack_get_tile(cb_id, index, (uint32_t*)p_tile));
ALWI void cb_release_tile(uint32_t cb_id) {
UNPACK(llk_unpack_release_tile(cb_id));
MATH(llk_math_release_tile(cb_id));
PACK(llk_pack_release_tile(cb_id));
}  // namespace ckernel

## matmul.h:
#define MM_THROTTLE 0
namespace ckernel {
ALWI void mm_init(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t out_cb_id, const uint32_t transpose = 0) {
UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose)));
PACK((llk_pack_init(out_cb_id)));
ALWI void matmul_tiles(
UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose)));
ALWI void mm_block_init(
UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));
UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));
}  // namespace ckernel

## tile_move_copy.h:
namespace ckernel {
ALWI void copy_tile_to_dst_init_short(uint32_t cbid, uint32_t transpose = 0) {
ALWI void copy_tile_init(uint32_t cbid) { copy_tile_to_dst_init_short(cbid); }
ALWI void copy_tile_to_dst_init_short_with_dt(uint32_t old_cbid, uint32_t new_cbid, uint32_t transpose = 0) {
copy_tile_to_dst_init_short(new_cbid, transpose);
* For the in_tile_index to be valid for this call, cb_wait_front(n) had to be
ALWI void copy_tile(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
}  // namespace ckernel

## common.h:
static FORCE_INLINE uint32_t get_arg_addr(int arg_idx) { return (uint32_t)&rta_l1_base[arg_idx]; }
static FORCE_INLINE uint32_t get_common_arg_addr(int arg_idx) { return (uint32_t)&crta_l1_base[arg_idx]; }
FORCE_INLINE T get_arg_val(int arg_idx) {
FORCE_INLINE T get_common_arg_val(int arg_idx) {
inline uint8_t get_absolute_logical_x() {
inline uint8_t get_absolute_logical_y() {
inline uint8_t get_relative_logical_x() {
inline uint8_t get_relative_logical_y() {

# TT-METAL HOST API

# SFPU_CHAIN EXAMPLE: sfpu_eltwise_chain

## Compute Kernel (compute.cpp):
```cpp
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
namespace NAMESPACE {
void MAIN {
    init_sfpu(src_cb_index, result_cb_index);
    tile_regs_acquire();
    // the first 0 in copy_tile is the index into the circular buffer
    cb_wait_front(src_cb_index, one_tile);
    cb_wait_front(ones_cb_index, one_tile);
    copy_tile(src_cb_index, /*offset*/ 0, /*register_offset*/ 0);
    copy_tile(ones_cb_index, /*offset*/ 0, /*register_offset*/ 1);
    exp_tile_init();
    exp_tile(0);  // exp(input)
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);  // exp(input) + 1
    log_tile_init();
    log_tile(0);  // log(exp(input) + 1)
    cb_reserve_back(result_cb_index, one_tile);
    pack_tile(0, result_cb_index);  // copy tile 0 from the registers to the CB
    cb_pop_front(src_cb_index, one_tile);
    cb_pop_front(ones_cb_index, one_tile);
    tile_regs_release();
    cb_push_back(result_cb_index, one_tile);
}  // namespace NAMESPACE
```

## Writer Kernel (writer.cpp):
```cpp
#include "dataflow_api.h"
#include <cstdint>
void kernel_main() {
    const uint32_t output_buffer_addr = get_arg_val<uint32_t>(0);
    cb_wait_front(result_cb_index, one_tile);
    const uint32_t l1_read_addr = get_read_ptr(result_cb_index);
    noc_async_write_tile(0, interleaved_accessor, l1_read_addr);
    noc_async_write_barrier();
    cb_pop_front(result_cb_index, one_tile);
```

## Reader Kernel (reader.cpp):
```cpp
#include "dataflow_api.h"
#include "tt-metalium/constants.hpp"
#include <cstdint>
#include <cstring>
void kernel_main() {
    const uint32_t input_buffer_addr = get_arg_val<uint32_t>(0);
    cb_reserve_back(src_cb_index, one_tile);
    const uint32_t l1_write_addr = get_write_ptr(src_cb_index);
    noc_async_read_tile(0, interleaved_accessor, l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(src_cb_index, one_tile);
    cb_reserve_back(ones_cb_index, one_tile);
    const uint32_t ones_l1_write_addr = get_write_ptr(ones_cb_index);
    cb_push_back(ones_cb_index, one_tile);
```

## Host Code (sfpu_eltwise_chain.cpp):
```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <cmath>
#include <random>
#include <cstdint>
#include <vector>
using namespace tt::tt_metal;
int main() {
    Program program = CreateProgram();
    KernelHandle reader_kernel_id = CreateKernel(
        tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});
    KernelHandle writer_kernel_id = CreateKernel(
        tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});
    CreateKernel(
        tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});
    SetRuntimeArgs(program, reader_kernel_id, core, {src_dram_buffer->address()});
    SetRuntimeArgs(program, writer_kernel_id, core, {dst_dram_buffer->address()});
```

# SINGLE_CORE EXAMPLE: add_2_integers_in_compute

## Compute Kernel (add_2_tiles.cpp):
```cpp
#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
namespace NAMESPACE {
void MAIN {
    cb_wait_front(cb_in0, 1);  // Unpack
    cb_wait_front(cb_in1, 1);  // Unpack
    tile_regs_acquire();  // Math
    pack_tile(0, cb_out0);  // Pack
    tile_regs_release();  // Pack
    cb_pop_front(cb_in0, 1);  // Unpack
    cb_pop_front(cb_in1, 1);  // Unpack
    cb_push_back(cb_out0, 1);  // Pack
}  // namespace NAMESPACE
```

## Reader Kernel (reader_binary_1_tile.cpp):
```cpp
#include <cstdint>
void kernel_main() {
    uint32_t in0_addr = get_arg_val<uint32_t>(0);
    uint32_t in1_addr = get_arg_val<uint32_t>(1);
    cb_reserve_back(cb_in0, 1);
    uint32_t cb_in0_addr = get_write_ptr(cb_in0);
    noc_async_read_tile(0, in0, cb_in0_addr);  // read
    noc_async_read_barrier();                  // wait until the read is done
    cb_push_back(cb_in0, 1);                   // mark the tile as ready.
    cb_reserve_back(cb_in1, 1);
    uint32_t cb_in1_addr = get_write_ptr(cb_in1);
    noc_async_read_tile(0, in1, cb_in1_addr);
    noc_async_read_barrier();
    cb_push_back(cb_in1, 1);
```

## Writer Kernel (writer_1_tile.cpp):
```cpp
#include <cstdint>
void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    cb_wait_front(cb_out0, 1);
    uint32_t cb_out0_addr = get_read_ptr(cb_out0);
    noc_async_write_tile(0, dst, cb_out0_addr);
    noc_async_write_barrier();  // This will wait until the write is done. As an alternative,
                                // noc_async_write_flushed() can be faster because it waits
                                // use noc_async_write_barrier() at least once at the end of
    cb_pop_front(cb_out0, 1);
```

## Host Code (add_2_integers_in_compute.cpp):
```cpp
#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "tt-metalium/constants.hpp"
#include <tt-metalium/distributed.hpp>
using namespace tt::tt_metal;
int main() {
    Program program = CreateProgram();
    KernelHandle binary_reader_kernel_id = CreateKernel(
    KernelHandle unary_writer_kernel_id = CreateKernel(
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
    SetRuntimeArgs(
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {});
    SetRuntimeArgs(program, unary_writer_kernel_id, core, {(uint32_t)dst_dram_buffer->address()});
```

# SINGLE_CORE EXAMPLE: eltwise_binary

## Compute Kernel (tiles_add.cpp):
```cpp
#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"
namespace NAMESPACE {
void MAIN {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        tile_regs_acquire();
        cb_reserve_back(cb_out0, 1);
        pack_tile(dst_reg, cb_out0);
        cb_push_back(cb_out0, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
        tile_regs_release();
}  // namespace NAMESPACE
```

## Writer Kernel (read_tiles.cpp):
```cpp
#include <cstdint>
void kernel_main() {
    uint32_t in0_addr = get_arg_val<uint32_t>(0);
    uint32_t in1_addr = get_arg_val<uint32_t>(1);
    uint32_t n_tiles = get_arg_val<uint32_t>(2);
        cb_reserve_back(cb_in0, 1);
        cb_reserve_back(cb_in1, 1);  // wait until we have 1 free slot. This blocks if the
        uint32_t cb_in0_addr = get_write_ptr(cb_in0);
        uint32_t cb_in1_addr = get_write_ptr(cb_in1);
        noc_async_read_tile(i, in0, cb_in0_addr);  // read the tile into the circular buffer
        noc_async_read_tile(i, in1, cb_in1_addr);  // We can overlap async reads and writes
        noc_async_read_barrier();  // Wait until tile reads are done
        cb_push_back(cb_in0, 1);
        cb_push_back(cb_in1, 1);  // mark the tiles as ready. From this point forward kernels
                                  // calling `cb_wait_front` will see this tile
```

## Writer Kernel (write_tile.cpp):
```cpp
#include <cstdint>
void kernel_main() {
    uint32_t c_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);
        cb_wait_front(cb_out0, 1);
        uint32_t cb_out0_addr = get_read_ptr(cb_out0);
        noc_async_write_tile(i, out0, cb_out0_addr);
        noc_async_write_barrier();  // This will wait until the write is done. As an alternative,
                                    // noc_async_write_flushed() can be faster because it waits
                                    // use noc_async_write_barrier() at least once at the end of
        cb_pop_front(cb_out0, 1);
```

## Host Code (eltwise_binary.cpp):
```cpp
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>
#include "tt-metalium/base_types.hpp"
using namespace tt::tt_metal;
int main(int /*argc*/, char** /*argv*/) {
        Program program = CreateProgram();
        auto reader = CreateKernel(
        auto writer = CreateKernel(
        auto compute = CreateKernel(
        SetRuntimeArgs(program, reader, core, {src0_dram_buffer->address(), src1_dram_buffer->address(), n_tiles});
        SetRuntimeArgs(program, writer, core, {dst_dram_buffer->address(), n_tiles});
        SetRuntimeArgs(program, compute, core, {n_tiles});
```

Generate high-quality, production-ready TT-Metal code following the patterns above.
```

## User Prompt

```
Generate a TT-Metal reader kernel for single-core diode current equation (I = isat × (exp(V/vj) - 1)).

Requirements:
- Read input tiles from DRAM to CB_0 and CB_1
- Follow single-core reader patterns from knowledge base
- Use NOC async reads with barriers
- Runtime args: src0_addr, src1_addr
- Filename: kernels/dataflow/reader_binary_1_tile.cpp

Generate only the .cpp code.
```

## Prompt Statistics

- System Prompt Length: 14,784 characters
- User Prompt Length: 377 characters
- Total Prompt Length: 15,161 characters
- Estimated Tokens: ~3,790 tokens

## RAG Sources Included

- SFPU Chain examples
- Single-core examples
