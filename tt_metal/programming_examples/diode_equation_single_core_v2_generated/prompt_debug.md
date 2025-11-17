# Prompt Debug

## System
```
You are an expert TT-Metal kernel developer for Tenstorrent hardware.

Target: diode_equation (diode current equation (I = isat × (exp(V/vj) - 1))), mode: single-core implementation.

Follow patterns from the retrieved examples and respect core-mode specific dataflow/compute structure.



## Relevant API Functions

```cpp
// Function: add_binary_tile
// Header: tt_metal/include/compute_kernel_api/eltwise_binary_sfpu.h
* | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
* | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
*/
// clang-format on
ALWI void add_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)

// Function: add_binary_tile_init
// Header: tt_metal/include/compute_kernel_api/eltwise_binary_sfpu.h
/**
* Please refer to documentation for any_init.
*/
ALWI void add_binary_tile_init()

// Function: add_unary_tile
// Header: tt_metal/include/compute_kernel_api/eltwise_unary/binop_with_scalar.h
* | param1         | fp32 value scalar encoded as uint32                                        | uint32_t | Must be less than the size of the DST register buffer       | True     |
*/
// clang-format on
ALWI void add_unary_tile(uint32_t idst, uint32_t param1)

// Function: add_unary_tile_int32
// Header: tt_metal/include/compute_kernel_api/eltwise_unary/binop_with_scalar.h
* | param1          | int32 value scalar encoded as uint32                                       | uint32_t | Must be less than the size of the DST register buffer | True     |
*/
// clang-format on
ALWI void add_unary_tile_int32(uint32_t idst, uint32_t param1)

// Function: binary_bitwise_tile_init
// Header: tt_metal/include/compute_kernel_api/binary_bitwise_sfpu.h
/**
* Please refer to documentation for any_init.
*/
ALWI void binary_bitwise_tile_init()

// Function: binop_with_scalar_tile_init
// Header: tt_metal/include/compute_kernel_api/eltwise_unary/binop_with_scalar.h
/**
* Please refer to documentation for any_init.
*/
ALWI void binop_with_scalar_tile_init()

// Function: bitwise_and_binary_tile
// Header: tt_metal/include/compute_kernel_api/binary_bitwise_sfpu.h
* | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
* | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
*/
// clang-format on
ALWI void bitwise_and_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)

// Function: bitwise_and_uint16_binary_tile
// Header: tt_metal/include/compute_kernel_api/binary_bitwise_sfpu.h
ALWI void bitwise_and_uint16_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)

// Function: bitwise_and_uint32_binary_tile
// Header: tt_metal/include/compute_kernel_api/binary_bitwise_sfpu.h
ALWI void bitwise_and_uint32_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)

// Function: bitwise_or_binary_tile
// Header: tt_metal/include/compute_kernel_api/binary_bitwise_sfpu.h
ALWI void bitwise_or_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)

// Function: bitwise_or_uint16_binary_tile
// Header: tt_metal/include/compute_kernel_api/binary_bitwise_sfpu.h
ALWI void bitwise_or_uint16_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)

// Function: bitwise_or_uint32_binary_tile
// Header: tt_metal/include/compute_kernel_api/binary_bitwise_sfpu.h
ALWI void bitwise_or_uint32_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)

// Function: bitwise_xor_binary_tile
// Header: tt_metal/include/compute_kernel_api/binary_bitwise_sfpu.h
ALWI void bitwise_xor_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)

// Function: bitwise_xor_uint16_binary_tile
// Header: tt_metal/include/compute_kernel_api/binary_bitwise_sfpu.h
ALWI void bitwise_xor_uint16_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)

// Function: bitwise_xor_uint32_binary_tile
// Header: tt_metal/include/compute_kernel_api/binary_bitwise_sfpu.h
ALWI void bitwise_xor_uint32_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)

// Function: cb_pop_front
// Header: tt_metal/include/compute_kernel_api/cb_api.h
* | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
* | ntiles    | The number of tiles to be popped     | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
*/
// clang-format on
ALWI void cb_pop_front(uint32_t cbid, uint32_t ntiles)

// Function: cb_push_back
// Header: tt_metal/include/compute_kernel_api/cb_api.h
* | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
* | ntiles    | The number of tiles to be pushed     | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
*/
// clang-format on
ALWI void cb_push_back(uint32_t cbid, uint32_t ntiles)

// Function: cb_reserve_back
// Header: tt_metal/include/compute_kernel_api/cb_api.h
* | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
* | ntiles    | The number of free tiles to wait for | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
*/
// clang-format on
ALWI void cb_reserve_back(uint32_t cbid, uint32_t ntiles)

// Function: cb_wait_front
// Header: tt_metal/include/compute_kernel_api/cb_api.h
* | cb_id     | The index of the cirular buffer (CB) | uint32_t | 0 to 31                                                                                           | True     |
* | ntiles    | The number of tiles to wait for      | uint32_t | It must be less or equal than the size of the CB (the total number of tiles that fit into the CB) | True     |
* */
// clang-format on
ALWI void cb_wait_front(uint32_t cbid, uint32_t ntiles)

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

// Function: div_unary_tile
// Header: tt_metal/include/compute_kernel_api/eltwise_unary/binop_with_scalar.h
ALWI void div_unary_tile(uint32_t idst, uint32_t param1)

// Function: exp_tile
// Header: tt_metal/include/compute_kernel_api/eltwise_unary/exp.h
ALWI void exp_tile(uint32_t idst, int vector_mode = (int)

// Function: exp_tile_init
// Header: tt_metal/include/compute_kernel_api/eltwise_unary/exp.h
* true.
*
*/
ALWI void exp_tile_init()

// Function: init_sfpu
// Header: tt_metal/include/compute_kernel_api/eltwise_unary/eltwise_unary.h
ALWI void init_sfpu(uint32_t icb, uint32_t ocb)

// Function: mask_posinf_tile
// Header: tt_metal/include/compute_kernel_api/mask.h
ALWI void mask_posinf_tile(uint32_t idst_data, uint32_t idst2_mask)

// Function: mask_tile
// Header: tt_metal/include/compute_kernel_api/mask.h
* | dst_mask_index | The index of the tile in DST REG for the mask                              | uint32_t   | Must be less than the acquired size of DST REG        | True     |
* | data_format    | The format of the data and mask (supports Float16, Float16_b, and Int32)   | DataFormat | Must be a valid data format                           | False    |
*/
// clang-format on
ALWI void mask_tile(uint32_t idst_data, uint32_t idst2_mask, DataFormat data_format = DataFormat::Float16_b)

// Function: mask_tile_init
// Header: tt_metal/include/compute_kernel_api/mask.h
ALWI void mask_tile_init()

// Function: mul_binary_tile
// Header: tt_metal/include/compute_kernel_api/eltwise_binary_sfpu.h
ALWI void mul_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)

// Function: mul_tiles
// Header: tt_metal/include/compute_kernel_api/eltwise_binary.h
* | in1_tile_index | The index of tile B within the second CB                 | uint32_t | Must be less than the size of the CB           | True     |
* | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t | Must be less than the acquired size of DST REG | True     |
*/
// clang-format on
ALWI void mul_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)

// Function: mul_unary_tile
// Header: tt_metal/include/compute_kernel_api/eltwise_unary/binop_with_scalar.h
ALWI void mul_unary_tile(uint32_t idst, uint32_t param1)

// Function: pack_tile
// Header: tt_metal/include/compute_kernel_api/pack.h
* | Function   | output_tile_index| The index of the tile in the output CB to copy to | uint32_t | Must be less than the size of the CB                 | False    |
*/
// clang-format on
ALWI void pack_tile(uint32_t ifrom_dst, uint32_t icb, std::uint32_t output_tile_index = 0)

// Function: rsub_unary_tile
// Header: tt_metal/include/compute_kernel_api/eltwise_unary/binop_with_scalar.h
ALWI void rsub_unary_tile(uint32_t idst, uint32_t param1)

// Function: sub_binary_tile
// Header: tt_metal/include/compute_kernel_api/eltwise_binary_sfpu.h
ALWI void sub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)

// Function: sub_tiles
// Header: tt_metal/include/compute_kernel_api/eltwise_binary.h
* | in1_tile_index | The index of tile B within the second CB                 | uint32_t | Must be less than the size of the CB           | True     |
* | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t | Must be less than the acquired size of DST REG | True     |
*/
// clang-format on
ALWI void sub_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)

// Function: sub_unary_tile
// Header: tt_metal/include/compute_kernel_api/eltwise_unary/binop_with_scalar.h
ALWI void sub_unary_tile(uint32_t idst, uint32_t param1)

// Function: sub_unary_tile_int32
// Header: tt_metal/include/compute_kernel_api/eltwise_unary/binop_with_scalar.h
* | param1          | int32 value scalar encoded as uint32                                       | uint32_t | Must be less than the size of the DST register buffer | True     |
*/
// clang-format on
ALWI void sub_unary_tile_int32(uint32_t idst, uint32_t param1)

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

## User
```
Generate TT-Metal kernels for diode current equation (I = isat × (exp(V/vj) - 1)) (single-core).

Requirements:
- Emit exactly three separate code blocks in your response
- Label each block clearly: COMPUTE, READER, WRITER
- Circular buffers: CB_0 for V, CB_1 for vj, CB_2 for constant isat, CB_16 for output
- Compute kernel: Implement the formula: I = isat × (exp(V/vj) - 1). Mathematical steps: divide V by vj, exponentiate result, subtract 1, multiply by isat. Use appropriate SFPU operations from the examples. Follow the pattern: initialize operations, wait for inputs, acquire registers, perform computation, pack result, release registers
- Reader kernel: Read V and vj tiles from DRAM. Initialize CB_2 with the constant isat value. Use noc_async_read with barriers
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
