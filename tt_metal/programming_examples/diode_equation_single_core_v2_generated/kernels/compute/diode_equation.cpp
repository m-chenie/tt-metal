// COMPUTE KERNEL: diode_equation.cpp
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"

namespace NAMESPACE {
void MAIN {
    // Compile time args
    constexpr uint32_t v_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t vj_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t isat_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t ones_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t result_cb_index = get_compile_time_arg_val(4);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Initialize the SFPU
    init_sfpu(v_cb_index, result_cb_index);

    // Wait for the SFPU to have registers available for us to use during
    // the computation.
    tile_regs_acquire();

    // Wait for data to show up in the circular buffer and copy it from
    // the circular buffer to registers so the SFPU can use it.
    cb_wait_front(v_cb_index, one_tile);
    cb_wait_front(vj_cb_index, one_tile);
    cb_wait_front(isat_cb_index, one_tile);
    cb_wait_front(ones_cb_index, one_tile);
    copy_tile(v_cb_index, /*offset*/ 0, /*register_offset*/ 0);
    copy_tile(vj_cb_index, /*offset*/ 0, /*register_offset*/ 1);
    copy_tile(isat_cb_index, /*offset*/ 0, /*register_offset*/ 2);
    copy_tile(ones_cb_index, /*offset*/ 0, /*register_offset*/ 3);

    //
    // Fused operations
    //
    // Compute the division of V by vj using the SFPU.
    div_binary_tile_init();
    div_binary_tile(0, 1, 4);  // V/vj

    // Compute the exponentiation of the result using the SFPU.
    exp_tile_init();
    exp_tile(4);  // exp(V/vj)

    // Subtract 1 from the result using the SFPU.
    sub_binary_tile_init();
    sub_binary_tile(4, 3, 4);  // exp(V/vj) - 1

    // Multiply the result by isat using the SFPU.
    mul_binary_tile_init();
    mul_binary_tile(4, 2, 0);  // isat * (exp(V/vj) - 1)

    // Wait for result to be done and data stored back to the circular buffer
    tile_regs_commit();
    tile_regs_wait();

    // Reserve output tile
    cb_reserve_back(result_cb_index, one_tile);

    pack_tile(0, result_cb_index);  // copy tile 0 from the registers to the CB

    // We don't need the input tile anymore, mark it as consumed
    cb_pop_front(v_cb_index, one_tile);
    cb_pop_front(vj_cb_index, one_tile);
    cb_pop_front(isat_cb_index, one_tile);
    cb_pop_front(ones_cb_index, one_tile);

    // Done with the registers, we can release them for the next SFPU operation
    tile_regs_release();

    // Mark the tile as ready for the writer kernel to write to DRAM
    cb_push_back(result_cb_index, one_tile);
}
}  // namespace NAMESPACE
