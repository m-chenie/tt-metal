// READER KERNEL
#include "dataflow_api.h"
#include "tt-metalium/constants.hpp"

#include <cstdint>
#include <cstring>

void kernel_main() {
    // Runtime args
    const uint32_t voltage_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t vj_buffer_addr = get_arg_val<uint32_t>(1);

    // Compile time args
    constexpr uint32_t voltage_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t vj_cb_index = get_compile_time_arg_val(1);

    // Input data config
    const uint32_t voltage_data_tile_size_bytes = get_tile_size(voltage_cb_index);
    const uint32_t vj_data_tile_size_bytes = get_tile_size(vj_cb_index);
    constexpr auto interleaved_accessor_args = TensorAccessorArgs<1>();

    const auto voltage_accessor = TensorAccessor<tensor_accessor::DistributionSpec<1, 1>>(
        interleaved_accessor_args, voltage_buffer_addr, voltage_data_tile_size_bytes);
    const auto vj_accessor = TensorAccessor<tensor_accessor::DistributionSpec<1, 1>>(
        interleaved_accessor_args, vj_buffer_addr, vj_data_tile_size_bytes);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Read voltage data
    cb_reserve_back(voltage_cb_index, one_tile);
    const uint32_t voltage_l1_write_addr = get_write_ptr(voltage_cb_index);
    noc_async_read_tile(0, voltage_accessor, voltage_l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(voltage_cb_index, one_tile);

    // Read vj data
    cb_reserve_back(vj_cb_index, one_tile);
    const uint32_t vj_l1_write_addr = get_write_ptr(vj_cb_index);
    noc_async_read_tile(0, vj_accessor, vj_l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(vj_cb_index, one_tile);
}
