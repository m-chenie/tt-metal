// SPDX-License-Identifier: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    // Retrieve the destination address from the kernel arguments
    uint32_t dst_addr = get_arg_val<uint32_t>(0);

    // Define the circular buffer index for the output
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    // Get the tile size used in the circular buffer
    const uint32_t tile_size_bytes = get_tile_size(cb_out0);

    // Set up the address generator for the output buffer
    const InterleavedAddrGenFast<true> dst = {
        .bank_base_address = dst_addr,         // Base address of the output buffer
        .page_size = tile_size_bytes,          // Size of a buffer page
        .data_format = DataFormat::Float16_b,  // Data format of the buffer
    };

    // Ensure there is a tile available in the circular buffer
    cb_wait_front(cb_out0, 1);
    uint32_t cb_out0_addr = get_read_ptr(cb_out0);

    // Perform an asynchronous write of the tile to DRAM
    noc_async_write_tile(0, dst, cb_out0_addr);
    // Wait for the write operation to complete
    noc_async_write_barrier();

    // Mark the tile as consumed in the circular buffer
    cb_pop_front(cb_out0, 1);
}
