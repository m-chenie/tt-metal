// SPDX-License-Identifier: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    // The circular buffer that we are going to read from and write to DRAM
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_out);

    // Address of the output buffer
    const InterleavedAddrGenFast<true> dst = {
        .bank_base_address = dst_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };

    // Loop through the assigned tiles and write them from circular buffer to DRAM
    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        uint32_t current_tile_id = start_tile_id + tile_id;

        // Wait for the result tile to be available in the circular buffer
        cb_wait_front(cb_out, 1);
        uint32_t cb_out_addr = get_read_ptr(cb_out);

        // Write the tile to DRAM
        noc_async_write_tile(current_tile_id, dst, cb_out_addr);
        noc_async_write_barrier();

        // Mark the tile as consumed
        cb_pop_front(cb_out, 1);
    }
}
