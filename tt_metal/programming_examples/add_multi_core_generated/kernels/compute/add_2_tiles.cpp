// SPDX-License-Identifier: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

using std::uint32_t;

namespace NAMESPACE {
/**
 * @brief Main kernel function for multi-core addition (C = A + B).
 *
 * This function performs element-wise addition of two matrices using tiles.
 * It initializes the addition operation and sets up circular buffers for input and output.
 * For each output tile (indexed by i), it:
 *   - Acquires the destination buffer.
 *   - Waits for input tiles to be available in the circular buffers.
 *   - Performs a tile-wise addition using `add_tiles`.
 *   - Pops the used tiles from the input buffers.
 *   - Reserves space in the output buffer, packs the result tile, and pushes it to the output buffer.
 *   - Releases the destination buffer.
 *
 * Runtime arguments:
 *   - num_output_tiles: Number of output tiles to produce.
 *
 * Circular buffers:
 *   - cb_in0: Input buffer for matrix A tiles.
 *   - cb_in1: Input buffer for matrix B tiles.
 *   - cb_out: Output buffer for result tiles.
 */
void MAIN {
    uint32_t num_output_tiles = get_arg_val<uint32_t>(0);  // number of output tiles to produce

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    // Setup the addition operation and specify the input and output circular buffers.
    binary_op_init_common(cb_in0, cb_in1, cb_out);
    add_tiles_init(cb_in0, cb_in1);

    // Loop over the number of output tiles to produce
    for (uint32_t i = 0; i < num_output_tiles; ++i) {
        // Acquire tile registers for the output tile
        tile_regs_acquire();

        // Wait for input tiles to be available in the circular buffers
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        // Perform the addition for the current tile
        add_tiles(cb_in0, cb_in1, 0, 0, 0);

        // Commit and wait for the registers to be populated with the results
        tile_regs_commit();
        tile_regs_wait();

        // Reserve space in the output circular buffer for the result tile
        cb_reserve_back(cb_out, 1);
        // Pack the result tile into the output circular buffer
        pack_tile(0, cb_out);
        // Mark the output tile as ready
        cb_push_back(cb_out, 1);

        // Release the tile registers
        tile_regs_release();

        // Pop the used input tiles from the circular buffers
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
}
}  // namespace NAMESPACE
