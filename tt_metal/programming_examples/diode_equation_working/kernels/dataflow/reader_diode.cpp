// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <cstdint>

void kernel_main() {
    // Read parameters from the kernel arguments
    uint32_t v_addr = get_arg_val<uint32_t>(0);
    uint32_t isat_addr = get_arg_val<uint32_t>(1);
    uint32_t vj_addr = get_arg_val<uint32_t>(2);

    // Define circular buffer indices for input tiles
    constexpr uint32_t cb_v = tt::CBIndex::c_0;     // Voltage
    constexpr uint32_t cb_vj = tt::CBIndex::c_1;    // Junction voltage
    constexpr uint32_t cb_isat = tt::CBIndex::c_2;  // Saturation current
    constexpr uint32_t cb_ones = tt::CBIndex::c_3;  // Ones tensor

    // Determine the tile size used in the circular buffers
    const uint32_t tile_size_bytes = get_tile_size(cb_v);

    // Setup address generators for the input buffers
    const InterleavedAddrGenFast<true> v_gen = {
        .bank_base_address = v_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };
    const InterleavedAddrGenFast<true> isat_gen = {
        .bank_base_address = isat_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };
    const InterleavedAddrGenFast<true> vj_gen = {
        .bank_base_address = vj_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };

    // Constants
    constexpr uint32_t one_tile = 1;

    // Read voltage (V) data
    cb_reserve_back(cb_v, one_tile);
    uint32_t l1_write_addr_v = get_write_ptr(cb_v);
    noc_async_read_tile(0, v_gen, l1_write_addr_v);
    noc_async_read_barrier();
    cb_push_back(cb_v, one_tile);

    // Read saturation current (isat) data
    cb_reserve_back(cb_isat, one_tile);
    uint32_t l1_write_addr_isat = get_write_ptr(cb_isat);
    noc_async_read_tile(0, isat_gen, l1_write_addr_isat);
    noc_async_read_barrier();
    cb_push_back(cb_isat, one_tile);

    // Read junction voltage (vj) data
    cb_reserve_back(cb_vj, one_tile);
    uint32_t l1_write_addr_vj = get_write_ptr(cb_vj);
    noc_async_read_tile(0, vj_gen, l1_write_addr_vj);
    noc_async_read_barrier();
    cb_push_back(cb_vj, one_tile);

    // Generate ones tensor (filled with 1.0 values)
    cb_reserve_back(cb_ones, one_tile);
    uint32_t l1_write_addr_ones = get_write_ptr(cb_ones);

    // Fill with ones - we need to fill the tile with 1.0 values
    // This is a simple approach - writing 1.0 in Float16_b format
    volatile tt_l1_ptr uint16_t* ones_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_ones);
    const uint16_t one_f16b = 0x3C00;  // 1.0 in Float16_b format
    const uint32_t tile_size_in_f16 = tile_size_bytes / sizeof(uint16_t);

    for (uint32_t i = 0; i < tile_size_in_f16; i++) {
        ones_ptr[i] = one_f16b;
    }

    cb_push_back(cb_ones, one_tile);
}
