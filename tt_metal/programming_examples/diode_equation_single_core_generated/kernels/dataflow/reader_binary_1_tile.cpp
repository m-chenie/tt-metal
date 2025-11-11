// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt-metalium/constants.hpp"

#include <cstdint>
#include <cstring>

void kernel_main() {
    // Runtime args
    const uint32_t src0_addr = get_arg_val<uint32_t>(0);
    const uint32_t src1_addr = get_arg_val<uint32_t>(1);

    // Compile time args
    constexpr uint32_t src0_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t src1_cb_index = get_compile_time_arg_val(1);

    // Input data config
    const uint32_t input_data_tile_size_bytes = get_tile_size(src0_cb_index);
    constexpr auto interleaved_accessor_args = TensorAccessorArgs<2>();
    const auto interleaved_accessor_src0 =
        TensorAccessor(interleaved_accessor_args, src0_addr, input_data_tile_size_bytes);
    const auto interleaved_accessor_src1 =
        TensorAccessor(interleaved_accessor_args, src1_addr, input_data_tile_size_bytes);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Read input data for V (voltage)
    cb_reserve_back(src0_cb_index, one_tile);
    const uint32_t l1_write_addr_src0 = get_write_ptr(src0_cb_index);
    noc_async_read_tile(0, interleaved_accessor_src0, l1_write_addr_src0);
    noc_async_read_barrier();
    cb_push_back(src0_cb_index, one_tile);

    // Read input data for isat (saturation current)
    cb_reserve_back(src1_cb_index, one_tile);
    const uint32_t l1_write_addr_src1 = get_write_ptr(src1_cb_index);
    noc_async_read_tile(0, interleaved_accessor_src1, l1_write_addr_src1);
    noc_async_read_barrier();
    cb_push_back(src1_cb_index, one_tile);
}
