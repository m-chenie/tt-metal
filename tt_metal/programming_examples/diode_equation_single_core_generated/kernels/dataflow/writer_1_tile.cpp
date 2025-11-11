// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr uint32_t result_cb_index = 16;  // CB_16

    // Constants
    constexpr uint32_t one_tile = 1;

    // Wait for the result tile to be available in the circular buffer
    cb_wait_front(result_cb_index, one_tile);

    // Get the read pointer for the result tile
    const uint32_t l1_read_addr = get_read_ptr(result_cb_index);

    // Write the result tile to DRAM using NOC async write
    noc_async_write_tile(0, dst_addr, l1_read_addr);

    // Wait for the write operation to complete
    noc_async_write_barrier();

    // Mark the tile as consumed in the circular buffer
    cb_pop_front(result_cb_index, one_tile);
}
