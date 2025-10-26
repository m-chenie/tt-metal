// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    constexpr auto cb_in0 = tt::CBIndex::c_0;    // Circular buffer for input A
    constexpr auto cb_in1 = tt::CBIndex::c_1;    // Circular buffer for input B
    constexpr auto cb_out0 = tt::CBIndex::c_16;  // Circular buffer for output C

    // Initialize the binary operation for subtraction
    binary_op_init_common(cb_in0, cb_in1, cb_out0);  // Unpack, Math, Pack
    sub_tiles_init(cb_in0, cb_in1);                  // Unpack, Math

    // Wait for a tile to be ready in the input circular buffers
    cb_wait_front(cb_in0, 1);  // Unpack
    cb_wait_front(cb_in1, 1);  // Unpack

    // Acquire tile registers to perform the subtraction
    tile_regs_acquire();  // Math

    // Perform the subtraction of tiles from cb_in0 and cb_in1
    // Store the result in cb_out0
    sub_tiles(cb_in0, cb_in1, 0, 0, 0);  // Unpack, Math

    // Signal the packer to commit the results
    tile_regs_commit();  // Math

    // Wait for the packer to complete
    tile_regs_wait();  // Pack

    // Pack the result from tile registers to the output circular buffer
    pack_tile(0, cb_out0);  // Pack

    // Release the tile registers
    tile_regs_release();  // Pack

    // Pop the processed tiles from the input circular buffers
    cb_pop_front(cb_in0, 1);  // Unpack
    cb_pop_front(cb_in1, 1);  // Unpack

    // Push the result tile to the output circular buffer
    cb_push_back(cb_out0, 1);  // Pack
}
}  // namespace NAMESPACE
