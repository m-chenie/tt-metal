// SPDX-License-Identifier: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    // Runtime arguments for work distribution
    uint32_t src0_addr = get_arg_val<uint32_t>(0);      // Base address for input A
    uint32_t src1_addr = get_arg_val<uint32_t>(1);      // Base address for input B
    uint32_t num_tiles = get_arg_val<uint32_t>(2);      // Number of tiles to read
    uint32_t start_tile_id = get_arg_val<uint32_t>(3);  // Starting tile ID for this core

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;

    // Get the tile size used in the circular buffers
    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);

    // Address generators for the input buffers
    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, src0_addr, in0_tile_bytes);

    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, src1_addr, in1_tile_bytes);

    // Loop through the assigned tiles and read them into the circular buffers
    for (uint32_t tile_id = start_tile_id; tile_id < start_tile_id + num_tiles; ++tile_id) {
        // Read tile from input A
        {
            cb_reserve_back(cb_id_in0, 1);
            uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            noc_async_read_tile(tile_id, a, l1_write_addr_in0);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, 1);
        }

        // Read tile from input B
        {
            cb_reserve_back(cb_id_in1, 1);
            uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
            noc_async_read_tile(tile_id, b, l1_write_addr_in1);
            noc_async_read_barrier();
            cb_push_back(cb_id_in1, 1);
        }
    }
}
