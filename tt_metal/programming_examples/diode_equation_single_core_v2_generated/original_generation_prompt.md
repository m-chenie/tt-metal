# Original Generation Prompt

## Kernel Generation

### System
```
You are an expert TT-Metal kernel developer for Tenstorrent hardware.

Target: diode_equation (diode current equation (I = isat × (exp(V/vj) - 1))), mode: single-core implementation.

Follow patterns from the retrieved examples and respect core-mode specific dataflow/compute structure.



CRITICAL CONSTRAINTS FOR SFPU OPERATIONS:

- ALL computation MUST use SFPU operations (element-wise on DST registers)

- DO NOT create DRAM buffers for scalar constants - initialize them directly in circular buffer in the reader kernel

- Follow the sfpu_eltwise_chain pattern for constant initialization in reader kernel



CIRCULAR BUFFER TILE SEMANTICS:

- Each tile in a circular buffer is TILE_HW elements (typically 1024 for 32x32)

- If a CB has N tiles, write N * TILE_HW elements total (N complete tiles)

- Example: CB with 3 tiles = write elements [0..TILE_HW-1] for tile 0, [TILE_HW..2*TILE_HW-1] for tile 1, [2*TILE_HW..3*TILE_HW-1] for tile 2

- In compute kernel: copy_tile(cb_id, tile_index, dst_reg) where tile_index selects which of the N tiles

- When cb_wait_front(cb_id, N) and cb_pop_front(cb_id, N), the N must match the number of tiles you're actually using



INPUT CONFIGURATION:

- Variable inputs (from DRAM): V

- Constant inputs (initialize in reader kernel using float_to_bfloat16):

  * Vj = 1.0

  * Isat = 0.026

  * ones = 1.0



CIRCULAR BUFFER LAYOUT (create all CB in host code):

- CB_0: V (variable input from DRAM) [1 tile(s)]

- CB_1: Vj (constant = 1.0, initialized in reader kernel) [1 tile(s)]

- CB_2: Isat (constant = 0.026, initialized in reader kernel) [1 tile(s)]

- CB_3: ones (constant = 1.0, initialized in reader kernel) [1 tile(s)]

- CB_4: output result [1 tile(s)]



## Relevant API Functions

```cpp
// Function: cb_pop_front
// Header: tt_metal/include/compute_kernel_api/cb_api.h
* | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
* | ntiles    | The number of tiles to be popped     | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
*/
// clang-format on
ALWI void cb_pop_front(uint32_t cbid, uint32_t ntiles)

// Header: tt_metal/hw/inc/dataflow_api.h
* | num_tiles | The number of tiles to be popped      | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
*/
// clang-format on
void cb_pop_front(int32_t operand, int32_t num_pages)

// Header: tt_metal/hw/inc/dataflow_api.h
* A non-blocking call that tells the caller if the specified number of pages are available in the specified circular
* buffer (CB). This call is used by the consumer of the CB to see if the prodcuers has fill the CB with at least the
* specified number of tiles. Important note: in case multiple calls of cb_wait_front(n) are issued without a paired
* cb_pop_front() call, n is expected to be incremented by the user to be equal to a cumulative total of tiles. Example:
* 4 calls of cb_wait_front(8) followed by
a cb_pop_front(32)

// Header: tt_metal/hw/inc/dataflow_api.h
* A non-blocking call that tells the caller if the specified number of pages are available in the specified circular
* buffer (CB). This call is used by the consumer of the CB to see if the prodcuers has fill the CB with at least the
* specified number of tiles. Important note: in case multiple calls of cb_wait_front(n) are issued without a paired
* cb_pop_front() call, n is expected to be incremented by the user to be equal to a cumulative total of tiles. Example:
* 4 calls of cb_wait_front(8) followed by
a cb_pop_front(32)

// Function: cb_push_back
// Header: tt_metal/include/compute_kernel_api/cb_api.h
* | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
* | ntiles    | The number of tiles to be pushed     | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
*/
// clang-format on
ALWI void cb_push_back(uint32_t cbid, uint32_t ntiles)

// Header: tt_metal/hw/inc/dataflow_api.h
* | num_tiles | The number of tiles to be pushed      | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
*/
// clang-format on
void cb_push_back(const int32_t operand, const int32_t num_pages)

// Function: cb_reserve_back
// Header: tt_metal/include/compute_kernel_api/cb_api.h
* | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
* | ntiles    | The number of free tiles to wait for | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
*/
// clang-format on
ALWI void cb_reserve_back(uint32_t cbid, uint32_t ntiles)

// Header: tt_metal/hw/inc/dataflow_api.h
* | num_tiles | The number of free tiles to wait for  | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
*/
// clang-format on
void cb_reserve_back(int32_t operand, int32_t num_pages)

// Function: cb_wait_front
// Header: tt_metal/include/compute_kernel_api/cb_api.h
* | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
* | ntiles    | The number of tiles to wait for      | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
* */
// clang-format on
ALWI void cb_wait_front(uint32_t cbid, uint32_t ntiles)

// Header: tt_metal/hw/inc/dataflow_api.h
// clang-format off
/**
* A non-blocking call that tells the caller if the specified number of pages are available in the specified circular
* buffer (CB). This call is used by the consumer of the CB to see if the prodcuers has fill the CB with at least the
* specified number of tiles. Important note: in case multiple calls
of cb_wait_front(n)

// Header: tt_metal/hw/inc/dataflow_api.h
* A non-blocking call that tells the caller if the specified number of pages are available in the specified circular
* buffer (CB). This call is used by the consumer of the CB to see if the prodcuers has fill the CB with at least the
* specified number of tiles. Important note: in case multiple calls of cb_wait_front(n) are issued without a paired
* cb_pop_front() call, n is expected to be incremented by the user to be equal to a cumulative total of tiles. Example:
* 4 calls
of cb_wait_front(8)

// Header: tt_metal/hw/inc/dataflow_api.h
// clang-format off
/**
* A non-blocking call that tells the caller if the specified number of pages are available in the specified circular
* buffer (CB). This call is used by the consumer of the CB to see if the prodcuers has fill the CB with at least the
* specified number of tiles. Important note: in case multiple calls
of cb_wait_front(n)

// Header: tt_metal/hw/inc/dataflow_api.h
* A non-blocking call that tells the caller if the specified number of pages are available in the specified circular
* buffer (CB). This call is used by the consumer of the CB to see if the prodcuers has fill the CB with at least the
* specified number of tiles. Important note: in case multiple calls of cb_wait_front(n) are issued without a paired
* cb_pop_front() call, n is expected to be incremented by the user to be equal to a cumulative total of tiles. Example:
* 4 calls
of cb_wait_front(8)

// Header: tt_metal/hw/inc/dataflow_api.h
* | num_tiles | The number of tiles to wait for       | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) |          |
*/
// clang-format on
void cb_wait_front(int32_t operand, int32_t num_pages)

// Function: copy_tile
// Header: tt_metal/include/compute_kernel_api/tile_move_copy.h
* | in_tile_index  | The index of the tile to copy from the input CB   | uint32_t  | Must be less than the size of the CB                | True     |
* | dst_tile_index | The index of the tile in the DST register         | uint32_t  | Must be less than the size of the DST register (16) | True     |
* */
// clang-format on
ALWI void copy_tile(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index)

// Function: div_binary_tile
// Header: tt_metal/include/compute_kernel_api/eltwise_binary_sfpu.h
ALWI void div_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)

// Function: div_binary_tile_init
// Header: tt_metal/include/compute_kernel_api/eltwise_binary_sfpu.h
ALWI void div_binary_tile_init()

// Function: exp_tile
// Header: tt_metal/include/compute_kernel_api/eltwise_unary/exp.h
ALWI void exp_tile(uint32_t idst, int vector_mode = (int)

// Function: exp_tile_init
// Header: tt_metal/include/compute_kernel_api/eltwise_unary/exp.h
* true.
*
*/
ALWI void exp_tile_init()

// Function: get_read_ptr
// Header: tt_metal/hw/inc/dataflow_api.h
* | operand   | The index of the circular buffer (CB) | uint32_t | 0 to 31     | True     |
*/
// clang-format on
uint32_t get_read_ptr(uint32_t operand)

// Header: tt_metal/hw/inc/dataflow_api.h
uint32_t get_read_ptr()

// Function: get_tile_size
// Header: tt_metal/hw/inc/dataflow_api.h
// this API is used by both the reader and writer side of the CB
// it uses unpack_src_format, but because unpack_src_format == pack_dst_format, we can use either
int32_t get_tile_size(const std::int32_t operand)

// Header: tt_metal/hw/inc/dataflow_api.h
uint32_t get_tile_size()

// Function: get_write_ptr
// Header: tt_metal/hw/inc/dataflow_api.h
* | operand   | The index of the circular buffer (CB) | uint32_t | 0 to 31     | True     |
*/
// clang-format on
uint32_t get_write_ptr(uint32_t operand)

// Header: tt_metal/hw/inc/dataflow_api.h
uint32_t get_write_ptr()

// Function: init_sfpu
// Header: tt_metal/include/compute_kernel_api/eltwise_unary/eltwise_unary.h
ALWI void init_sfpu(uint32_t icb, uint32_t ocb)

// Function: mul_binary_tile
// Header: tt_metal/include/compute_kernel_api/eltwise_binary_sfpu.h
ALWI void mul_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)

// Function: mul_binary_tile_init
// Header: tt_metal/include/compute_kernel_api/eltwise_binary_sfpu.h
ALWI void mul_binary_tile_init()

// Function: noc_async_read_barrier
// Header: tt_metal/hw/inc/dataflow_api.h
* | Argument | Description                          | Type     | Valid Range | Required |
* |----------|--------------------------------------|----------|-------------|----------|
* | noc      | Which NOC to query on                | uint8_t  | 0 or 1      | False    |
*/
void noc_async_read_barrier(uint8_t noc = noc_index)

// Function: noc_async_read_tile
// Header: tt_metal/hw/inc/dataflow_api.h
*/
// clang-format on
void noc_async_read_tile(
    const uint32_t id,
    const InterleavedAddrGen<DRAM>& addrgen,
    uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index)

// Header: tt_metal/hw/inc/dataflow_api.h
*/
// clang-format on
void noc_async_read_tile(
    const uint32_t id,
    const TensorAccessor<DSpec>& addrgen,
    uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index)

// Header: tt_metal/hw/inc/dataflow_api.h
*/
// clang-format on
void noc_async_read_tile(
    const uint32_t id,
    const InterleavedAddrGenFast<DRAM, tile_hw>& addrgen,
    uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index)

// Function: noc_async_write_barrier
// Header: tt_metal/hw/inc/dataflow_api.h
* |----------|--------------------------------------|----------|-------------|----------|
* | noc      | Which NOC to query on                | uint8_t  | 0 or 1      | False    |
*/
void noc_async_write_barrier(uint8_t noc = noc_index)

// Function: noc_async_write_tile
// Header: tt_metal/hw/inc/dataflow_api.h
* Refer to template <typename AddrGen> noc_async_write_page for a generic implementation and more details.
*/
void noc_async_write_tile(
    const uint32_t id, const InterleavedAddrGen<DRAM>& addrgen, uint32_t src_local_l1_addr, uint8_t noc = noc_index)

// Header: tt_metal/hw/inc/dataflow_api.h
* | tile_hw (template parameter) | Tile height x width | uint32_t  | Any uint32_t number | True     |
*/
void noc_async_write_tile(
    const uint32_t id,
    const InterleavedAddrGenFast<DRAM, tile_hw>& addrgen,
    uint32_t src_local_l1_addr,
    uint8_t noc = noc_index)

// Header: tt_metal/hw/inc/dataflow_api.h
*/
// clang-format on
void noc_async_write_tile(
    const uint32_t id, const TensorAccessor<DSpec>& addrgen, uint32_t src_local_l1_addr, uint8_t noc = noc_index)

// Function: pack_tile
// Header: tt_metal/include/compute_kernel_api/pack.h
* | Function   | output_tile_index| The index of the tile in the output CB to copy to | uint32_t | Must be less than the size of the CB                 | False    |
*/
// clang-format on
ALWI void pack_tile(uint32_t ifrom_dst, uint32_t icb, std::uint32_t output_tile_index = 0)

// Function: sub_binary_tile
// Header: tt_metal/include/compute_kernel_api/eltwise_binary_sfpu.h
ALWI void sub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)

// Function: sub_binary_tile_init
// Header: tt_metal/include/compute_kernel_api/eltwise_binary_sfpu.h
* Please refer to documentation for any_init.
*/
ALWI void sub_binary_tile_init()

// Function: tile_regs_acquire
// Header: tt_metal/include/compute_kernel_api/reg_api.h
* Acquire an exclusive lock on the DST register for the MATH thread.
* This register is an array of 16 tiles of 32x32 elements each.
* This is a blocking function, i.e. this function will wait until the lock is acquired.
*/
ALWI void tile_regs_acquire()

// Function: tile_regs_commit
// Header: tt_metal/include/compute_kernel_api/reg_api.h
/**
* Release lock on DST register by MATH thread. The lock had to be previously acquired with tile_regs_acquire.
*/
ALWI void tile_regs_commit()

// Function: tile_regs_release
// Header: tt_metal/include/compute_kernel_api/reg_api.h
/**
* Release lock on DST register by PACK thread. The lock had to be previously acquired with tile_regs_wait.
*/
ALWI void tile_regs_release()

// Function: tile_regs_wait
// Header: tt_metal/include/compute_kernel_api/reg_api.h
* Acquire an exclusive lock on the DST register for the PACK thread.
* It waits for the MATH thread to commit the DST register.
* This is a blocking function, i.e. this function will wait until the lock is acquired.
*/
ALWI void tile_regs_wait()

```



## Compute Kernel Examples

# Source: /home/m48chen/tt-metal/tt_metal/programming_examples/sfpu_eltwise_chain/kernels/compute/compute.cpp
```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"

namespace NAMESPACE {
void MAIN {
    // Compile time args
    constexpr uint32_t src_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t ones_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t result_cb_index = get_compile_time_arg_val(2);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Initialize the SFPU
    init_sfpu(src_cb_index, result_cb_index);

    // Wait for the SFPU to have registers available for us to use during
    // the computation.
    tile_regs_acquire();

    // Wait for data to show up in the circular buffer and copy it from
    // the circular buffer to registers so the SFPU can use it.
    // the first 0 in copy_tile is the index into the circular buffer
    // and the second 0 is the offset into the registers. This case
    // we are copying the 0th tile from the source data circular buffer to the 0th tile
    // in the registers and 0th tile from the ones tile to the 1st tile in the registers.
    cb_wait_front(src_cb_index, one_tile);
    cb_wait_front(ones_cb_index, one_tile);
    copy_tile(src_cb_index, /*offset*/ 0, /*register_offset*/ 0);
    copy_tile(ones_cb_index, /*offset*/ 0, /*register_offset*/ 1);

    //
    // Fused operations
    //
    // Compute the softplus of the tile using the SFPU.
    // *_init() - Telling the SFPU to perform given operation. This is required each time we
    // switch to a different SFPU operation.
    exp_tile_init();
    exp_tile(0);  // exp(input)

    add_binary_tile_init();
    add_binary_tile(0, 1, 0);  // exp(input) + 1

    log_tile_init();
    log_tile(0);  // log(exp(input) + 1)

    // Wait for result to be done and data stored back to the circular buffer
    tile_regs_commit();
    tile_regs_wait();

    // Reserve output tile
    cb_reserve_back(result_cb_index, one_tile);

    pack_tile(0, result_cb_index);  // copy tile 0 from the registers to the CB

    // We don't need the input tile anymore, mark it as consumed
    cb_pop_front(src_cb_index, one_tile);
    cb_pop_front(ones_cb_index, one_tile);

    // Done with the registers, we can release them for the next SFPU operation
    tile_regs_release();

    // Mark the tile as ready for the writer kernel to write to DRAM
    cb_push_back(result_cb_index, one_tile);
}
}  // namespace NAMESPACE

```


## Reader Kernel Examples

# Source: /home/m48chen/tt-metal/tt_metal/programming_examples/sfpu_eltwise_chain/kernels/dataflow/reader.cpp
```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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
    const uint32_t input_buffer_addr = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr uint32_t src_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t ones_cb_index = get_compile_time_arg_val(1);

    // Input data config
    const uint32_t input_data_tile_size_bytes = get_tile_size(src_cb_index);
    constexpr auto interleaved_accessor_args = TensorAccessorArgs<2>();
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

    // Create tile with ones
    cb_reserve_back(ones_cb_index, one_tile);
    const uint32_t ones_l1_write_addr = get_write_ptr(ones_cb_index);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ones_l1_write_addr);
    for (uint32_t i = 0; i < tt::constants::TILE_HW; i++) {
        ptr[i] = float_to_bfloat16(1.0f);
    }
    cb_push_back(ones_cb_index, one_tile);
}

```


## Writer Kernel Examples

# Source: /home/m48chen/tt-metal/tt_metal/programming_examples/sfpu_eltwise_chain/kernels/dataflow/writer.cpp
```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t output_buffer_addr = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr uint32_t result_cb_index = get_compile_time_arg_val(0);

    // Input data config
    const uint32_t output_data_tile_size_bytes = get_tile_size(result_cb_index);
    constexpr auto interleaved_accessor_args = TensorAccessorArgs<1>();
    const auto interleaved_accessor =
        TensorAccessor(interleaved_accessor_args, output_buffer_addr, output_data_tile_size_bytes);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Save output data
    cb_wait_front(result_cb_index, one_tile);
    const uint32_t l1_read_addr = get_read_ptr(result_cb_index);
    noc_async_write_tile(0, interleaved_accessor, l1_read_addr);
    noc_async_write_barrier();
    cb_pop_front(result_cb_index, one_tile);
}

```


Generate concise, correct TT-Metal code following these patterns.
```

### User
```
Generate TT-Metal kernels for diode current equation (I = isat × (exp(V/vj) - 1)) (single-core).

Requirements:
- Emit exactly three separate code blocks in your response
- Label each block clearly: COMPUTE, READER, WRITER
- Circular buffer layout (MUST follow exactly):
  * CB_0: V (variable input from DRAM) [1 tile(s)]
  * CB_1: Vj (constant = 1.0, initialized in reader kernel) [1 tile(s)]
  * CB_2: Isat (constant = 0.026, initialized in reader kernel) [1 tile(s)]
  * CB_3: ones (constant = 1.0, initialized in reader kernel) [1 tile(s)]
  * CB_4: output result [1 tile(s)]
- Compute kernel: Implement the formula: I = isat × (exp(V/vj) - 1). Mathematical steps: divide V by vj, exponentiate result, subtract 1, multiply by isat. DO NOT initialize constants in compute kernel. REMINDER: cb_wait_front/cb_pop_front count must match the number of tiles in the CB (e.g., if CB has 3 tiles, use cb_wait_front(cb_id, 3) and cb_pop_front(cb_id, 3)). Use appropriate SFPU operations from the examples. Follow the pattern: initialize operations, wait for inputs, acquire registers, perform computation, pack result, release registers
- Reader kernel: Read V from DRAM into CB_0. Initialize constant tiles in reader kernel using float_to_bfloat16 pattern:   * Vj = 1.0 (in appropriate CB as specified above)   * Isat = 0.026 (in appropriate CB as specified above)   * ones = 1.0 (in appropriate CB as specified above)
REMINDER: When writing N tiles to a CB:   - Each tile is TILE_HW elements (1024 for 32x32)   - Write tile 0 at ptr[0..TILE_HW-1], tile 1 at ptr[TILE_HW..2*TILE_HW-1], etc.   - Call cb_reserve_back(cb_id, N) and cb_push_back(cb_id, N) to reserve/push N tiles. Use noc_async_read with barriers
- Writer kernel: Write output tiles from CB_16 to DRAM using noc_async_write with barriers
- Study the provided examples to identify the correct API functions and usage patterns
- Follow TT-Metal conventions: cb_wait/reserve/push/pop discipline, NOC barriers, proper includes

Expected output format:
```cpp
// COMPUTE KERNEL: diode_equation.cpp
[compute kernel code here]
```

```cpp
// READER KERNEL: reader_binary_1_tile.cpp
[reader kernel code here]
```

```cpp
// WRITER KERNEL: writer_1_tile.cpp
[writer kernel code here]
```
```


## Host Code Generation

### System
```
You are an expert TT-Metal host code developer for Tenstorrent hardware.

Target: diode_equation (diode current equation (I = isat × (exp(V/vj) - 1))), mode: single-core implementation.

Generate correct, modern TT-Metal host code following the canonical structure and examples.



## Complete Host Code Examples

Study these examples to understand the full workflow:


### Example: /home/m48chen/tt-metal/tt_metal/programming_examples/sfpu_eltwise_chain/sfpu_eltwise_chain.cpp


```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>

#include <cmath>
#include <random>
#include <cstdint>
#include <vector>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

/**
 * @brief Computes the softplus activation function element-wise on input vector
 *
 * The softplus function is defined as: softplus(x) = log(1 + exp(x))
 * This is a smooth approximation to the ReLU function that outputs positive values.
 *
 * @param src_vec Input vector containing bfloat16 values to apply softplus to
 * @param result_vec Output vector where computed softplus values will be stored
 *                   Must be the same size as src_vec
 *
 * @throws TT_FATAL if input and output vectors have different sizes
 *
 * @note This is a reference implementation used for validation/testing purposes
 * @note The function uses std::log1p for numerical stability
 */
void golden_softplus(const std::vector<bfloat16>& src_vec, std::vector<bfloat16>& result_vec) {
    TT_FATAL(src_vec.size() == result_vec.size(), "Input and output vectors must be the same size");
    for (size_t i = 0; i < src_vec.size(); ++i) {
        result_vec[i] = bfloat16(std::log1p(std::exp(static_cast<float>(src_vec[i]))));  // Softplus function
    }
}

/**
 * @brief Calculates the Pearson Correlation Coefficient (PCC) between two bfloat16 vectors.
 *
 * This function computes the linear correlation coefficient between two vectors of bfloat16 values,
 * which measures the strength and direction of the linear relationship between the two datasets.
 * The PCC value ranges from -1 to 1, where:
 * - 1 indicates a perfect positive linear relationship
 * - 0 indicates no linear relationship
 * - -1 indicates a perfect negative linear relationship
 *
 * @param vec_a First input vector of bfloat16 values
 * @param vec_b Second input vector of bfloat16 values (must be same size as vec_a)
 *
 * @return float The Pearson correlation coefficient between the two input vectors
 *
 * @note The function assumes both input vectors have the same size
 * @note bfloat16 values are converted to float for calculations to maintain precision
 */
inline float check_bfloat16_vector_pcc(const std::vector<bfloat16>& vec_a, const std::vector<bfloat16>& vec_b) {
    // Calculate the mean of x and y values
    float x_mean = 0.0f;
    float y_mean = 0.0f;

    for (size_t i = 0; i < vec_a.size(); i++) {
        x_mean += static_cast<float>(vec_a[i]);
        y_mean += static_cast<float>(vec_b[i]);
    }

    x_mean /= vec_a.size();
    y_mean /= vec_b.size();

    // Calculate the covariance and standard deviation of x and y values
    float covariance = 0.0f;
    float x_stddev = 0.0f;
    float y_stddev = 0.0f;

    for (size_t i = 0; i < vec_a.size(); i++) {
        float x_diff = static_cast<float>(vec_a[i]) - x_mean;
        float y_diff = static_cast<float>(vec_b[i]) - y_mean;

        covariance += x_diff * y_diff;
        x_stddev += x_diff * x_diff;
        y_stddev += y_diff * y_diff;
    }

    covariance /= vec_a.size();
    x_stddev /= vec_a.size();
    y_stddev /= vec_b.size();

    // Calculate the correlation coefficient
    float correlation_coefficient_ = covariance / (std::sqrt(x_stddev) * std::sqrt(y_stddev));
    return correlation_coefficient_;
}

int main() {
    // Device setup
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);

    // Device command queue and program setup
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    // Core range setup
    constexpr CoreCoord core = {0, 0};

    // Input data preparation
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.f, 1.0f);

    // Fill the source vector with random values
    std::vector<bfloat16> src_vec(constants::TILE_HW);
    for (bfloat16& v : src_vec) {
        v = bfloat16(dist(rng));
    }

    // Calculate golden function results on CPU
    std::vector<bfloat16> golden_vec(constants::TILE_HW, 0);
    golden_softplus(src_vec, golden_vec);

    // Tilize the input vectors to match the expected tiled layout for the device
    // The Tenstorrent hardware operates on data in 32x32 tiles rather than standard row-major format.
    // tilize_nfaces() converts the input matrices from row-major layout to the tiled layout expected by the device.
    // This transformation groups elements into 32x32 blocks and reorders them in memory so that each tile
    // (32x32 elements) is stored contiguously. This matches the native data access patterns of the matrix engine
    // and enables efficient operations on the accelerator.
    src_vec = tilize_nfaces(src_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // Dram buffer config
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * constants::TILE_HEIGHT * constants::TILE_WIDTH;
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = sizeof(bfloat16) * src_vec.size()};
    std::shared_ptr<distributed::MeshBuffer> src_dram_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());  // Input buffer
    std::shared_ptr<distributed::MeshBuffer> dst_dram_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());  // Output buffer

    // DRAM transfer
    distributed::EnqueueWriteMeshBuffer(cq, src_dram_buffer, src_vec, false);

    // L1 circular buffer setup
    constexpr uint32_t src_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src_config =
        CircularBufferConfig(single_tile_size, {{src_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src_config);

    constexpr uint32_t ones_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_ones_config =
        CircularBufferConfig(single_tile_size, {{ones_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ones_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_ones_config);

    constexpr uint32_t result_cb_index = CBIndex::c_2;
    CircularBufferConfig cb_result_config =
        CircularBufferConfig(single_tile_size, {{result_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(result_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_result_config);

    // Kernels setup
    // Data movement kernels
    std::vector<uint32_t> reader_compile_time_args = {src_cb_index, ones_cb_index};
    TensorAccessorArgs(*src_dram_buffer).append_to(reader_compile_time_args);
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "sfpu_eltwise_chain/kernels/dataflow/reader.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});
    std::vector<uint32_t> writer_compile_time_args = {result_cb_index};
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "sfpu_eltwise_chain/kernels/dataflow/writer.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});

    // Compute kernel
    std::vector<uint32_t> compute_compile_time_args = {src_cb_index, ones_cb_index, result_cb_index};
    CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "sfpu_eltwise_chain/kernels/compute/compute.cpp",
        core,
        tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});

    // Runtime args setup
    SetRuntimeArgs(program, reader_kernel_id, core, {src_dram_buffer->address()});
    SetRuntimeArgs(program, writer_kernel_id, core, {dst_dram_buffer->address()});

    // Program enqueue
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Data transfer back to host machine
    std::vector<bfloat16> result_vec(constants::TILE_HW, 0);
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

    // Reverse the tilization to get the result in the row-major format that the CPU expects
    result_vec = untilize_nfaces(result_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // Calculate the Pearson correlation coefficient (PCC) between the golden vector and the result vector
    // This is a measure of how similar the two vectors are.
    // A PCC close to 1 indicates that the two vectors are very similar.
    const float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
    fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
    TT_FATAL(pearson > 0.999, "PCC not high enough. Result PCC: {}, Expected PCC: 0.999", pearson);

    mesh_device->close();
}

```


### Example: /home/m48chen/tt-metal/tt_metal/programming_examples/eltwise_binary/eltwise_binary.cpp


```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>
#include "tt-metalium/base_types.hpp"

using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main(int /*argc*/, char** /*argv*/) {
    bool pass = true;

    // clang-format off
    try {
        // Create a 1x1 mesh on device 0. The same API scales to multi-device meshes.
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        // Submit work via a mesh command queue: data uploads/downloads and program execution.
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        // A program is a collection of kernels. Note that unlike OpenCL/CUDA where every core must run the
        // same kernel at a given time. Metalium allows you to run different kernels on different cores
        // simultaneously.
        distributed::MeshWorkload workload;
        // Execute across this device range. Here it spans the whole mesh (1x1).
        auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram();

        // This example program will only use 1 Tensix core. So we set the core to {0, 0}.
        constexpr CoreCoord core = {0, 0};

        // Define some constants that will be used throughout the program.
        // * Processing 64 tiles
        // * Each tile is 32x32 elements
        // * Each element is a bfloat16 (2 bytes)
        constexpr uint32_t n_tiles = 64;
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;

        // Create 3 DRAM-backed mesh buffers: two inputs (src0, src1) and one output (dst).
        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = tile_size_bytes, //The page size of the buffer in bytes. Unlike the `loopback` example, we
                                          // need the page size to be the same as the tile size for a large portion of the NoC transfer APIs to work.
            .buffer_type = BufferType::DRAM}; // This is a DRAM buffer.
        distributed::ReplicatedBufferConfig buffer_config{
            .size = n_tiles * tile_size_bytes // Total bytes per device (replicated across the mesh).
        };

        auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        // Each handle represents a mesh-wide replicated buffer; on a unit mesh this is a single device allocation.

        // Initialize the input buffers with random data. For this example, src0 is a random vector of bfloat16 values
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        std::vector<bfloat16> a_data(elements_per_tile * n_tiles);
        for(auto& val : a_data) {
            val = bfloat16(distribution(rng));
        }

        // ... and src1 is a vector of bfloat16 values initialized to -1.0f.
        constexpr float val_to_add = -1.0f;
        std::vector<bfloat16> b_data(elements_per_tile * n_tiles, bfloat16(val_to_add));

        // Upload host vectors into the mesh buffers.
        distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, a_data, false);
        distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, b_data, false);

        // Create 3 circular buffers. Think them like pipes moving data from one core to another. cb_src0 and cb_src1 are used to
        // move data from the reader kernel to the compute kernel. cb_dst is used to move data from the compute kernel to the writer
        // kernel. Each circular buffer is made up of 2 tiles. Thus when one tile is pushed and being used by the receiving end, the
        // sending end can get the next piece of data ready to be pushed. Overlapping the operations. Leading to better performance.
        // However there is a trade off, The more tiles in a circular buffer, the more memory is used. And Circular buffers are
        // backed by L1(SRAM) memory and L1 is a precious resource.
        // The hardware supports up to 32 circular buffers and they all act the same.
        constexpr uint32_t tiles_per_cb = 2;
        tt::CBIndex src0_cb_index = tt::CBIndex::c_0;
        CreateCircularBuffer(program, core, CircularBufferConfig(
            /*total_size=*/tiles_per_cb * tile_size_bytes,                    // The total size of the circular buffer in bytes
            /*data_format_spec=*/{{src0_cb_index, tt::DataFormat::Float16_b}})// The circular buffer index and data format it'll hold
            .set_page_size(src0_cb_index, tile_size_bytes));                  // Since we will be sending one tile at a time, we set
                                                                              // the page size to the tile size (and thus
                                                                              // total_size / page_size = tiles_per is the number of
                                                                              // entries in the circular buffer)
        tt::CBIndex src1_cb_index = tt::CBIndex::c_1;
        CreateCircularBuffer(program, core, CircularBufferConfig(
            /*total_size=*/tiles_per_cb * tile_size_bytes,
            /*data_format_spec=*/{{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, tile_size_bytes));
        tt::CBIndex dst_cb_index = tt::CBIndex::c_16;
        CreateCircularBuffer(program, core, CircularBufferConfig(
            /*total_size=*/tiles_per_cb * tile_size_bytes,
            /*data_format_spec=*/{{dst_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(dst_cb_index, tile_size_bytes));

        // Create the reader, writer and compute kernels. The kernels do the following:
        // * Reader: Reads data from the DRAM buffer and pushes it into the circular buffer.
        // * Compute: Waits for data to be available in the circular buffer, pops it, adds the two inputs together and pushes the result
        //   into the output circular buffer.
        // * Writer: Waits for data to be available in the output circular buffer, pops it and writes it back into DRAM.
        // These kernels work together to form a pipeline. The reader reads data from the DRAM buffer and makes them available in the
        // compute kernel. The compute kernel does math and pushes the result into the writer kernel. The writer kernel writes the result
        // back to DRAM.
        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);
        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/dataflow/read_tiles.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_compile_time_args});
        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
        auto writer = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/dataflow/write_tile.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_compile_time_args});
        auto compute = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/compute/tiles_add.cpp",
            core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4});   // There's different math fidelity modes (for the tensor engine)
                                                                // that trade off performance for accuracy. HiFi4 is the most accurate
                                                                // mode. The other modes are HiFi3, HiFi2, HiFi1 and LoFi. The
                                                                // difference between them is the number of bits used during computation.

        // Set the runtime arguments for the kernels. This also registers
        // the kernels with the program.
        SetRuntimeArgs(program, reader, core, {src0_dram_buffer->address(), src1_dram_buffer->address(), n_tiles});
        SetRuntimeArgs(program, writer, core, {dst_dram_buffer->address(), n_tiles});
        SetRuntimeArgs(program, compute, core, {n_tiles});

        // We have setup the program. Now we queue the kernel for execution. The final argument is set to false. This indicates
        // to Metalium that the operation is non-blocking. The function is allowed to return upon the kernel being queued. We must
        // ensure that the kernel is finished before we read the output buffer. This is done by calling distributed::Finish(cq) which waits until
        // all operations in the command queue are finished. This is equivalent to calling EnqueueMeshWorkload(cq, program, true); telling
        // Metalium to wait until the program is finished before returning.
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        // Equivalently:
        // distributed::EnqueueMeshWorkload(cq, workload, true);

        // Read the output buffer (from shard at mesh coordinate {0,0} on a unit mesh) and validate.
        std::vector<bfloat16> result_vec;
        distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

        constexpr float eps = 1e-2f; // loose tolerance because of the nature of bfloat16
        TT_FATAL(result_vec.size() == a_data.size(), "Result vector size mismatch");
        for (size_t i = 0; i < result_vec.size(); ++i) {
            const float expected = static_cast<float>(a_data[i]) + val_to_add;
            const float actual = static_cast<float>(result_vec[i]);

            if (std::abs(expected - actual) > eps) {
                pass = false;
                fmt::print(stderr, "Result mismatch at index {}: expected {}, got {}\n", i, expected, actual);
            }
        }

        // Finally, we close the device.
        pass &= mesh_device->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());

        throw;
    }
    // clang-format on

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}

```


## CMakeLists.txt Examples

Use these patterns for proper library linking:


### Example: /home/m48chen/tt-metal/tt_metal/programming_examples/sfpu_eltwise_chain/CMakeLists.txt


```cmake
cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_sfpu_eltwise_chain)

add_executable(metal_example_sfpu_eltwise_chain)
target_sources(metal_example_sfpu_eltwise_chain PRIVATE sfpu_eltwise_chain.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_sfpu_eltwise_chain PUBLIC TT::Metalium)

```


### Example: /home/m48chen/tt-metal/tt_metal/programming_examples/eltwise_binary/CMakeLists.txt


```cmake
cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_eltwise_binary)

add_executable(metal_example_eltwise_binary)
target_sources(metal_example_eltwise_binary PRIVATE eltwise_binary.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_eltwise_binary PUBLIC TT::Metalium)

```


CRITICAL REQUIREMENTS:

- Use ONLY angle bracket includes: `#include <tt-metalium/host_api.hpp>` NOT quotes

- Use `distributed::MeshDevice` NOT `Device`

- Use `distributed::MeshBuffer` NOT `Buffer`

- All distributed APIs require `distributed::` namespace prefix

- CMakeLists.txt MUST include: find_package(TT-Metalium) and target_link_libraries(...TT::Metalium)
```

### User
```
Generate complete host code (.cpp file) AND CMakeLists.txt for diode current equation (I = isat × (exp(V/vj) - 1)) (single-core).

Requirements:
- Use correct headers: `#include <tt-metalium/host_api.hpp>` with angle brackets
- Use `distributed::MeshDevice::create_unit_mesh()` for device setup
- Create DRAM buffers using `distributed::MeshBuffer::create()`
- CRITICAL: Allocate ALL circular buffers in host code using CreateCircularBuffer():
  * CB_0: size=single_tile_size bytes, format=Float16_b
  * CB_1: size=single_tile_size bytes, format=Float16_b
  * CB_2: size=single_tile_size bytes, format=Float16_b
  * CB_3: size=single_tile_size bytes, format=Float16_b
  * CB_4: size=single_tile_size bytes, format=Float16_b
- DO NOT create DRAM buffers for constants or initialize constant data in host code
- Constants will be initialized directly in L1 by the reader kernel
- ONLY create DRAM buffer for variable input: V
- Compile and launch the three kernels (reader, compute, writer) with SetRuntimeArgs
- Enqueue program using `distributed::EnqueueMeshWorkload()`
- Add CPU golden validation with PCC check
- Use proper tilize/untilize for data conversion

Output format - provide TWO code blocks:

```cpp
// HOST CODE: diode_equation_single_v2.cpp
[complete host code here]
```

```cmake
# CMakeLists.txt
[complete CMakeLists.txt with proper linking]
```

Ensure CMakeLists.txt includes find_package(TT-Metalium) and target_link_libraries with TT::Metalium.
```
