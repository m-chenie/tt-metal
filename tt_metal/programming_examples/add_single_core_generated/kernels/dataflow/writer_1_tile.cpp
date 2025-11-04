// SPDX-License-Identifier: Apache-2.0
//
// TT-Metal Writer Kernel for Single-Core Addition
// This kernel writes one tile from circular buffer CB_16 to DRAM.

#include <cstdint>

void kernel_main() {
    // Retrieve the destination address from the runtime argument
    uint32_t dst_addr = get_arg_val<uint32_t>(0);

    // Define the circular buffer index for the output
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

    // Obtain the tile size used in the circular buffer
    const uint32_t tile_size_bytes = get_tile_size(cb_out0);

    // Configure the address generator for the output buffer
    const InterleavedAddrGenFast<true> dst = {
        .bank_base_address = dst_addr,         // Base address of the output buffer
        .page_size = tile_size_bytes,          // Size of a buffer page
        .data_format = DataFormat::Float16_b,  // Data format of the buffer
    };

    // Ensure there is a tile available in the circular buffer
    cb_wait_front(cb_out0, 1);

    // Get the read pointer for the circular buffer
    uint32_t cb_out0_addr = get_read_ptr(cb_out0);

    // Perform an asynchronous write of the tile to DRAM
    noc_async_write_tile(0, dst, cb_out0_addr);

    // Wait for the write operation to complete
    noc_async_write_barrier();

    // Mark the tile as consumed in the circular buffer
    cb_pop_front(cb_out0, 1);
}
