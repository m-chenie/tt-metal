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
    constexpr uint32_t src_cb_index = 0;      // CB_0 for V
    constexpr uint32_t isat_cb_index = 1;     // CB_1 for isat
    constexpr uint32_t vj_cb_index = 1;       // CB_1 for vj
    constexpr uint32_t result_cb_index = 16;  // CB_16 for output

    // Constants
    constexpr uint32_t one_tile = 1;

    // Initialize the SFPU
    init_sfpu(src_cb_index, result_cb_index);

    // Wait for the SFPU to have registers available for us to use during
    // the computation.
    tile_regs_acquire();

    // Wait for data to show up in the circular buffer and copy it from
    // the circular buffer to registers so the SFPU can use it.
    cb_wait_front(src_cb_index, one_tile);
    cb_wait_front(isat_cb_index, one_tile);
    copy_tile(src_cb_index, /*offset*/ 0, /*register_offset*/ 0);
    copy_tile(isat_cb_index, /*offset*/ 0, /*register_offset*/ 1);

    // Compute V/vj
    div_binary_tile_init();
    div_binary_tile(0, 1, 2);  // V/vj

    // Compute exp(V/vj)
    exp_tile_init();
    exp_tile(2);  // exp(V/vj)

    // Compute exp(V/vj) - 1
    sub_binary_tile_init();
    sub_binary_tile(2, 1, 3);  // exp(V/vj) - 1, assuming 1 is in register 1

    // Compute isat * (exp(V/vj) - 1)
    mul_binary_tile_init();
    mul_binary_tile(3, 1, 0);  // isat * (exp(V/vj) - 1)

    // Wait for result to be done and data stored back to the circular buffer
    tile_regs_commit();
    tile_regs_wait();

    // Reserve output tile
    cb_reserve_back(result_cb_index, one_tile);

    pack_tile(0, result_cb_index);  // copy tile 0 from the registers to the CB

    // We don't need the input tile anymore, mark it as consumed
    cb_pop_front(src_cb_index, one_tile);
    cb_pop_front(isat_cb_index, one_tile);

    // Done with the registers, we can release them for the next SFPU operation
    tile_regs_release();

    // Mark the tile as ready for the writer kernel to write to DRAM
    cb_push_back(result_cb_index, one_tile);
}
}  // namespace NAMESPACE
