// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    // Define circular buffer indices for inputs and output
    constexpr auto cb_in0 = tt::CBIndex::c_0;    // Circular buffer for input A
    constexpr auto cb_in1 = tt::CBIndex::c_1;    // Circular buffer for input B
    constexpr auto cb_out0 = tt::CBIndex::c_16;  // Circular buffer for output C

    // Initialize the binary operation for addition
    binary_op_init_common(cb_in0, cb_in1, cb_out0);  // Initialize common settings for unpack, math, and pack cores
    add_tiles_init(cb_in0, cb_in1);                  // Initialize addition operation for unpack and math cores

    // Wait for a tile to be ready in the input circular buffers
    cb_wait_front(cb_in0, 1);  // Wait for input A to be ready
    cb_wait_front(cb_in1, 1);  // Wait for input B to be ready

    // Acquire tile registers to perform the addition
    tile_regs_acquire();  // Acquire registers for math operations

    // Perform the addition of the input tiles and store the result in the output tile
    add_tiles(cb_in0, cb_in1, 0, 0, 0);  // Add tiles from input A and B, store result in output C

    // Signal the packer to commit the result
    tile_regs_commit();  // Commit the result to the packer

    // Wait for the packer to complete the operation
    tile_regs_wait();  // Wait for the packer to finish

    // Pack the result from tile registers to the output circular buffer
    pack_tile(0, cb_out0);  // Pack the result into the output circular buffer

    // Release the tile registers
    tile_regs_release();  // Release the registers after packing

    // Pop the processed tiles from the input circular buffers
    cb_pop_front(cb_in0, 1);  // Remove the processed tile from input A
    cb_pop_front(cb_in1, 1);  // Remove the processed tile from input B

    // Push the result tile to the output circular buffer
    cb_push_back(cb_out0, 1);  // Mark the output tile as ready
}
}  // namespace NAMESPACE
