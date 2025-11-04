// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    // Read parameters from the kernel arguments
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);

    // Define circular buffer indices for input tiles
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;

    // Determine the tile size used in the circular buffers
    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    // Setup address generators for the input buffers
    const InterleavedAddrGenFast<true> src0 = {
        .bank_base_address = src0_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };
    const InterleavedAddrGenFast<true> src1 = {
        .bank_base_address = src1_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };

    // Read the first tile from DRAM into the first circular buffer
    cb_reserve_back(cb_in0, 1);
    uint32_t cb_in0_addr = get_write_ptr(cb_in0);
    noc_async_read_tile(0, src0, cb_in0_addr);
    noc_async_read_barrier();
    cb_push_back(cb_in0, 1);

    // Read the second tile from DRAM into the second circular buffer
    cb_reserve_back(cb_in1, 1);
    uint32_t cb_in1_addr = get_write_ptr(cb_in1);
    noc_async_read_tile(0, src1, cb_in1_addr);
    noc_async_read_barrier();
    cb_push_back(cb_in1, 1);
}
