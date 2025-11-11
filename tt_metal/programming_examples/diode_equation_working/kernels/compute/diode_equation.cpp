// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"

namespace NAMESPACE {
void MAIN {
    // Compile time args
    constexpr uint32_t V_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t vj_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t isat_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t ones_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t result_cb_index = get_compile_time_arg_val(4);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Initialize SFPU for unary+binary operations (like sfpu_eltwise_chain)
    init_sfpu(V_cb_index, result_cb_index);

    // Acquire tile registers
    tile_regs_acquire();

    // Load all inputs to registers
    cb_wait_front(V_cb_index, one_tile);
    cb_wait_front(vj_cb_index, one_tile);
    cb_wait_front(isat_cb_index, one_tile);
    cb_wait_front(ones_cb_index, one_tile);
    copy_tile(V_cb_index, /*offset*/ 0, /*register_offset*/ 0);     // V → R0
    copy_tile(vj_cb_index, /*offset*/ 0, /*register_offset*/ 1);    // vj → R1
    copy_tile(isat_cb_index, /*offset*/ 0, /*register_offset*/ 2);  // isat → R2
    copy_tile(ones_cb_index, /*offset*/ 0, /*register_offset*/ 3);  // ones → R3

    // Chain operations: I = isat * (exp(V/vj) - 1)
    div_binary_tile_init();
    div_binary_tile(0, 1, 0);  // R0 = V/vj

    exp_tile_init();
    exp_tile(0);  // R0 = exp(V/vj)

    sub_binary_tile_init();
    sub_binary_tile(0, 3, 0);  // R0 = exp(V/vj) - 1

    mul_binary_tile_init();
    mul_binary_tile(2, 0, 0);  // R0 = isat * (exp(V/vj) - 1)

    // Store final result
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(result_cb_index, one_tile);
    pack_tile(0, result_cb_index);

    // Mark input tiles as consumed
    cb_pop_front(V_cb_index, one_tile);
    cb_pop_front(vj_cb_index, one_tile);
    cb_pop_front(isat_cb_index, one_tile);
    cb_pop_front(ones_cb_index, one_tile);

    // Release tile registers
    tile_regs_release();

    // Push result to output buffer
    cb_push_back(result_cb_index, one_tile);
}
}  // namespace NAMESPACE
