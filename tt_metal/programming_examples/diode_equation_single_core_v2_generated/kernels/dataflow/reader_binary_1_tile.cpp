// READER KERNEL: reader_binary_1_tile.cpp
#include "dataflow_api.h"
#include "tt-metalium/constants.hpp"

#include <cstdint>
#include <cstring>

void kernel_main() {
    // Runtime args
    const uint32_t input_buffer_addr_V = get_arg_val<uint32_t>(0);
    const uint32_t input_buffer_addr_isat_vj = get_arg_val<uint32_t>(1);

    // Compile time args
    constexpr uint32_t src_cb_index = 0;   // CB_0 for V
    constexpr uint32_t isat_cb_index = 1;  // CB_1 for isat and vj

    // Input data config
    const uint32_t input_data_tile_size_bytes = get_tile_size(src_cb_index);
    constexpr auto interleaved_accessor_args = TensorAccessorArgs<2>();
    const auto interleaved_accessor_V =
        TensorAccessor(interleaved_accessor_args, input_buffer_addr_V, input_data_tile_size_bytes);
    const auto interleaved_accessor_isat_vj =
        TensorAccessor(interleaved_accessor_args, input_buffer_addr_isat_vj, input_data_tile_size_bytes);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Read input value data
    cb_reserve_back(src_cb_index, one_tile);
    const uint32_t l1_write_addr_V = get_write_ptr(src_cb_index);
    noc_async_read_tile(0, interleaved_accessor_V, l1_write_addr_V);
    noc_async_read_barrier();
    cb_push_back(src_cb_index, one_tile);

    // Read input isat and vj data
    cb_reserve_back(isat_cb_index, one_tile);
    const uint32_t l1_write_addr_isat_vj = get_write_ptr(isat_cb_index);
    noc_async_read_tile(0, interleaved_accessor_isat_vj, l1_write_addr_isat_vj);
    noc_async_read_barrier();
    cb_push_back(isat_cb_index, one_tile);
}
