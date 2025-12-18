// READER KERNEL: reader_binary_1_tile.cpp
#include "dataflow_api.h"
#include "tt-metalium/constants.hpp"

#include <cstdint>
#include <cstring>

/**
 * @brief Converts a 32-bit IEEE 754 float to 16-bit bfloat16 format.
 *
 * This function performs a simple truncation conversion from float32 to bfloat16
 * by extracting the upper 16 bits (sign, exponent, and upper 7 bits of mantissa)
 * of the IEEE 754 float representation. The lower 16 bits of the mantissa are
 * discarded, which may result in precision loss but maintains the same range
 * as float32.
 *
 * @param value The input 32-bit floating point value to convert
 * @return uint16_t The resulting 16-bit bfloat16 value in its binary representation
 *
 * @note This implementation uses simple truncation without rounding, which may
 *       introduce quantization errors for values that cannot be exactly
 *       represented in bfloat16 format - it is sufficient for this example.
 */
inline uint16_t float_to_bfloat16(float value) {
    uint32_t tmp;
    std::memcpy(&tmp, &value, sizeof(tmp));
    return static_cast<uint16_t>(tmp >> 16);
}

void kernel_main() {
    // Runtime args
    const uint32_t v_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t vj_buffer_addr = get_arg_val<uint32_t>(1);
    const uint32_t isat_buffer_addr = get_arg_val<uint32_t>(2);

    // Compile time args
    constexpr uint32_t v_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t vj_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t isat_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t ones_cb_index = get_compile_time_arg_val(3);

    // Input data config
    const uint32_t input_data_tile_size_bytes = get_tile_size(v_cb_index);
    constexpr auto interleaved_accessor_args = TensorAccessorArgs<2>();
    const auto v_accessor = TensorAccessor(interleaved_accessor_args, v_buffer_addr, input_data_tile_size_bytes);
    const auto vj_accessor = TensorAccessor(interleaved_accessor_args, vj_buffer_addr, input_data_tile_size_bytes);
    const auto isat_accessor = TensorAccessor(interleaved_accessor_args, isat_buffer_addr, input_data_tile_size_bytes);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Read input value data
    cb_reserve_back(v_cb_index, one_tile);
    const uint32_t v_l1_write_addr = get_write_ptr(v_cb_index);
    noc_async_read_tile(0, v_accessor, v_l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(v_cb_index, one_tile);

    cb_reserve_back(vj_cb_index, one_tile);
    const uint32_t vj_l1_write_addr = get_write_ptr(vj_cb_index);
    noc_async_read_tile(0, vj_accessor, vj_l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(vj_cb_index, one_tile);

    cb_reserve_back(isat_cb_index, one_tile);
    const uint32_t isat_l1_write_addr = get_write_ptr(isat_cb_index);
    noc_async_read_tile(0, isat_accessor, isat_l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(isat_cb_index, one_tile);

    // Create tile with ones
    cb_reserve_back(ones_cb_index, one_tile);
    const uint32_t ones_l1_write_addr = get_write_ptr(ones_cb_index);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ones_l1_write_addr);
    for (uint32_t i = 0; i < tt::constants::TILE_HW; i++) {
        ptr[i] = float_to_bfloat16(1.0f);
    }
    cb_push_back(ones_cb_index, one_tile);
}
