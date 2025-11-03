// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    constexpr auto cb_in0 = tt::CBIndex::c_0;    // Circular buffer for input A
    constexpr auto cb_in1 = tt::CBIndex::c_1;    // Circular buffer for input B
    constexpr auto cb_out0 = tt::CBIndex::c_16;  // Circular buffer for output C

    // Read the number of tiles to process from the kernel arguments
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // Initialize the binary operation for subtraction
    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    sub_tiles_init(cb_in0, cb_in1);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // Wait for a tile to be ready in the input circular buffers
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        // Acquire tile registers to perform the subtraction
        tile_regs_acquire();

        // Perform the subtraction on the current tile (always use register 0)
        sub_tiles(cb_in0, cb_in1, 0, 0, 0);

        // Commit the results to the packer
        tile_regs_commit();

        // Wait for the packer to finish
        tile_regs_wait();

        // Reserve space in output buffer and pack the result
        cb_reserve_back(cb_out0, 1);
        pack_tile(0, cb_out0);

        // Release the tile registers
        tile_regs_release();

        // Pop the processed tiles from the input circular buffers
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);

        // Push the result tile to the output circular buffer
        cb_push_back(cb_out0, 1);
    }
}
}  // namespace NAMESPACE
