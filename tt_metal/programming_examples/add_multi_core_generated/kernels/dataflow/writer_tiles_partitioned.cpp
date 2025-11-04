// SPDX-License-Identifier: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    // Runtime arguments to write data back into the output buffer.
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);  // number of output tiles to write
    uint32_t start_id = get_arg_val<uint32_t>(2);   // starting tile ID for output tiles

    constexpr uint32_t cb_id_out = tt::CBIndex::c_16;

    // Get tile size for calculations
    const uint32_t tile_bytes = get_tile_size(cb_id_out);

    // Configure the address generator for the output buffer using interleaved addressing
    const InterleavedAddrGenFast<true> dst = {
        .bank_base_address = dst_addr,         // Base address of the output buffer
        .page_size = tile_bytes,               // Size of a buffer page
        .data_format = DataFormat::Float16_b,  // Data format of the buffer
    };

    // Loop through the tile indices and write each tile to DRAM in order.
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        // Wait for the kernel to produce an output tile
        cb_wait_front(cb_id_out, 1);
        // Write the output tile to DRAM using the proper tile index
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        noc_async_write_tile(i, dst, l1_read_addr);
        noc_async_write_barrier();  // This will wait until the write is done. As an alternative,
                                    // noc_async_write_flushed() can be faster because it waits
                                    // until the write request is sent. In that case, you have to
                                    // use noc_async_write_barrier() at least once at the end of
                                    // data movement kernel to make sure all writes are done.
        cb_pop_front(cb_id_out, 1);
    }
}
