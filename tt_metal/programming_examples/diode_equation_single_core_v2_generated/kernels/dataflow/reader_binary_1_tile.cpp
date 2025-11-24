// READER KERNEL
#include "dataflow_api.h"
#include "tt-metalium/constants.hpp"

#include <cstdint>
#include <cstring>

inline uint16_t float_to_bfloat16(float value) {
    uint32_t tmp;
    std::memcpy(&tmp, &value, sizeof(tmp));
    return static_cast<uint16_t>(tmp >> 16);
}

void kernel_main() {
    // Runtime args
    const uint32_t input_buffer_addr = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr uint32_t src_cb_index = 0;
    constexpr uint32_t vj_cb_index = 1;
    constexpr uint32_t isat_cb_index = 2;
    constexpr uint32_t ones_cb_index = 3;

    // Input data config
    const uint32_t input_data_tile_size_bytes = get_tile_size(src_cb_index);
    constexpr auto interleaved_accessor_args = TensorAccessorArgs<1>();
    const auto interleaved_accessor =
        TensorAccessor(interleaved_accessor_args, input_buffer_addr, input_data_tile_size_bytes);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Read input value data
    cb_reserve_back(src_cb_index, one_tile);
    const uint32_t l1_write_addr = get_write_ptr(src_cb_index);
    noc_async_read_tile(0, interleaved_accessor, l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(src_cb_index, one_tile);

    // Create tile with constants
    cb_reserve_back(vj_cb_index, one_tile);
    const uint32_t vj_l1_write_addr = get_write_ptr(vj_cb_index);
    volatile tt_l1_ptr uint16_t* vj_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(vj_l1_write_addr);
    for (uint32_t i = 0; i < tt::constants::TILE_HW; i++) {
        vj_ptr[i] = float_to_bfloat16(1.0f);
    }
    cb_push_back(vj_cb_index, one_tile);

    cb_reserve_back(isat_cb_index, one_tile);
    const uint32_t isat_l1_write_addr = get_write_ptr(isat_cb_index);
    volatile tt_l1_ptr uint16_t* isat_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(isat_l1_write_addr);
    for (uint32_t i = 0; i < tt::constants::TILE_HW; i++) {
        isat_ptr[i] = float_to_bfloat16(1.0f);
    }
    cb_push_back(isat_cb_index, one_tile);

    cb_reserve_back(ones_cb_index, one_tile);
    const uint32_t ones_l1_write_addr = get_write_ptr(ones_cb_index);
    volatile tt_l1_ptr uint16_t* ones_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ones_l1_write_addr);
    for (uint32_t i = 0; i < tt::constants::TILE_HW; i++) {
        ones_ptr[i] = float_to_bfloat16(1.0f);
    }
    cb_push_back(ones_cb_index, one_tile);
}
