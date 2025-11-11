// SPDX-License-Identifier: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt-metalium/constants.hpp"

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t V_addr = get_arg_val<uint32_t>(0);
    const uint32_t vj_addr = get_arg_val<uint32_t>(1);
    const uint32_t isat_addr = get_arg_val<uint32_t>(2);

    // Compile time args
    constexpr uint32_t V_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t vj_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t isat_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t ones_cb_index = get_compile_time_arg_val(3);

    // Input data config
    const uint32_t input_data_tile_size_bytes = get_tile_size(V_cb_index);

    // TensorAccessorArgs for each input buffer - each append_to adds 1 value
    // After 4 CB indices: V at index 4, vj at index 5, isat at index 6
    constexpr auto V_accessor_args = TensorAccessorArgs<4>();
    constexpr auto vj_accessor_args = TensorAccessorArgs<5>();
    constexpr auto isat_accessor_args = TensorAccessorArgs<6>();

    const auto V_accessor = TensorAccessor(V_accessor_args, V_addr, input_data_tile_size_bytes);
    const auto vj_accessor = TensorAccessor(vj_accessor_args, vj_addr, input_data_tile_size_bytes);
    const auto isat_accessor = TensorAccessor(isat_accessor_args, isat_addr, input_data_tile_size_bytes);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Read voltage (V) data
    cb_reserve_back(V_cb_index, one_tile);
    const uint32_t l1_write_addr_V = get_write_ptr(V_cb_index);
    noc_async_read_tile(0, V_accessor, l1_write_addr_V);
    noc_async_read_barrier();
    cb_push_back(V_cb_index, one_tile);

    // Read junction voltage (vj) data
    cb_reserve_back(vj_cb_index, one_tile);
    const uint32_t l1_write_addr_vj = get_write_ptr(vj_cb_index);
    noc_async_read_tile(0, vj_accessor, l1_write_addr_vj);
    noc_async_read_barrier();
    cb_push_back(vj_cb_index, one_tile);

    // Read saturation current (isat) data
    cb_reserve_back(isat_cb_index, one_tile);
    const uint32_t l1_write_addr_isat = get_write_ptr(isat_cb_index);
    noc_async_read_tile(0, isat_accessor, l1_write_addr_isat);
    noc_async_read_barrier();
    cb_push_back(isat_cb_index, one_tile);

    // Generate ones tensor (filled with 1.0 values)
    cb_reserve_back(ones_cb_index, one_tile);
    uint32_t l1_write_addr_ones = get_write_ptr(ones_cb_index);

    // Fill with ones - we need to fill the tile with 1.0 values
    // This is a simple approach - writing 1.0 in Float16_b format
    volatile tt_l1_ptr uint16_t* ones_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_ones);
    const uint16_t one_f16b = 0x3C00;  // 1.0 in Float16_b format
    const uint32_t tile_size_in_f16 = input_data_tile_size_bytes / sizeof(uint16_t);

    for (uint32_t i = 0; i < tile_size_in_f16; i++) {
        ones_ptr[i] = one_f16b;
    }

    cb_push_back(ones_cb_index, one_tile);
}
