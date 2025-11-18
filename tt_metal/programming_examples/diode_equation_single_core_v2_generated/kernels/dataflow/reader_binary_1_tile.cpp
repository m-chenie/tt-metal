// READER KERNEL: reader_binary_1_tile.cpp
#include "dataflow_api.h"
#include "tt-metalium/constants.hpp"

#include <cstdint>
#include <cstring>

void kernel_main() {
    // Runtime args
    const uint32_t input_buffer_addr_V = get_arg_val<uint32_t>(0);
    const uint32_t input_buffer_addr_vj = get_arg_val<uint32_t>(1);

    // Compile time args
    constexpr uint32_t src_cb_index_V = 0;
    constexpr uint32_t src_cb_index_vj = 1;
    constexpr uint32_t src_cb_index_isat = 2;

    // Input data config
    const uint32_t input_data_tile_size_bytes = get_tile_size(src_cb_index_V);
    constexpr auto interleaved_accessor_args = TensorAccessorArgs<2>();
    const auto interleaved_accessor_V =
        TensorAccessor(interleaved_accessor_args, input_buffer_addr_V, input_data_tile_size_bytes);
    const auto interleaved_accessor_vj =
        TensorAccessor(interleaved_accessor_args, input_buffer_addr_vj, input_data_tile_size_bytes);

    // Constants
    constexpr uint32_t one_tile = 1;
    constexpr float isat_value = 1.0f;  // Replace with actual isat value

    // Read input value data
    cb_reserve_back(src_cb_index_V, one_tile);
    const uint32_t l1_write_addr_V = get_write_ptr(src_cb_index_V);
    noc_async_read_tile(0, interleaved_accessor_V, l1_write_addr_V);
    noc_async_read_barrier();

    cb_reserve_back(src_cb_index_vj, one_tile);
    const uint32_t l1_write_addr_vj = get_write_ptr(src_cb_index_vj);
    noc_async_read_tile(0, interleaved_accessor_vj, l1_write_addr_vj);
    noc_async_read_barrier();

    cb_reserve_back(src_cb_index_isat, one_tile);
    const uint32_t ones_l1_write_addr = get_write_ptr(src_cb_index_isat);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ones_l1_write_addr);
    for (uint32_t i = 0; i < tt::constants::TILE_HW; i++) {
        ptr[i] = float_to_bfloat16(isat_value);
    }

    cb_push_back(src_cb_index_V, one_tile);
    cb_push_back(src_cb_index_vj, one_tile);
    cb_push_back(src_cb_index_isat, one_tile);
}

inline uint16_t float_to_bfloat16(float value) {
    uint32_t tmp;
    std::memcpy(&tmp, &value, sizeof(tmp));
    return static_cast<uint16_t>(tmp >> 16);
}
