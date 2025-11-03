// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    // Read parameters from the kernel arguments
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t start_tile_id = get_arg_val<uint32_t>(5);
    uint32_t num_tiles = get_arg_val<uint32_t>(6);

    // The circular buffers to read the tiles into
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;

    // Get the tile size used in the circular buffers
    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    // Create address generators for the input buffers
    const InterleavedAddrGenFast<true> in0 = {
        .bank_base_address = src0_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };
    const InterleavedAddrGenFast<true> in1 = {
        .bank_base_address = src1_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };

    // Loop through the assigned tiles and read them from DRAM to circular buffers
    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        uint32_t current_tile_id = start_tile_id + tile_id;

        // Read first input tile (src0) into circular buffer 0
        cb_reserve_back(cb_in0, 1);
        uint32_t l1_write_addr_in0 = get_write_ptr(cb_in0);
        noc_async_read_tile(current_tile_id, in0, l1_write_addr_in0);
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);

        // Read second input tile (src1) into circular buffer 1
        cb_reserve_back(cb_in1, 1);
        uint32_t l1_write_addr_in1 = get_write_ptr(cb_in1);
        noc_async_read_tile(current_tile_id, in1, l1_write_addr_in1);
        noc_async_read_barrier();
        cb_push_back(cb_in1, 1);
    }
}
