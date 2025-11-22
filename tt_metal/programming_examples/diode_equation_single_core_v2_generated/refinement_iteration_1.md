# Iteration 1

## System
```
You are an expert TT-Metal developer fixing compilation errors.

This is iteration 1 of debugging.

================================================================================
PREVIOUS REQUEST (what you were asked to do):
================================================================================
# Prompt Debug

## Kernel Generation

### System
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

### User
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


## Host Code Generation

### System
```
You are an expert TT-Metal host code developer for Tenstorrent hardware.

Target: diode_equation (diode current equation (I = isat × (exp(V/vj) - 1))), mode: single-core implementation.

Generate correct, modern TT-Metal host code following the canonical structure and examples.



## Canonical Host Code Structure

Use this template as a guide for structuring your host code:

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// CORRECT HEADERS - use angle brackets for tt-metalium headers
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>

#include <cmath>
#include <random>
#include <cstdint>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;

int main() {
    // 1. DEVICE SETUP - use distributed::MeshDevice for modern TT-Metal
    std::shared_ptr<distributed::MeshDevice> mesh_device =
        distributed::MeshDevice::create_unit_mesh(0);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range =
        distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    // 2. CORE RANGE SETUP
    constexpr CoreCoord core = {0, 0};

    // 3. INPUT DATA PREPARATION
    std::vector<bfloat16> input_vec(constants::TILE_HW);
    // ... fill input_vec with data ...

    // Tilize input data for device
    input_vec = tilize_nfaces(input_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // 4. DRAM BUFFER CREATION - use distributed::MeshBuffer
    constexpr uint32_t single_tile_size =
        sizeof(bfloat16) * constants::TILE_HEIGHT * constants::TILE_WIDTH;
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM
    };
    distributed::ReplicatedBufferConfig buffer_config{
        .size = sizeof(bfloat16) * input_vec.size()
    };
    std::shared_ptr<distributed::MeshBuffer> input_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    // Write input data to device
    distributed::EnqueueWriteMeshBuffer(cq, input_buffer, input_vec, false);

    // 5. CIRCULAR BUFFER SETUP
    constexpr uint32_t cb_index = CBIndex::c_0;
    CircularBufferConfig cb_config =
        CircularBufferConfig(single_tile_size, {{cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_config);

    // 6. KERNEL CREATION
    std::vector<uint32_t> compile_time_args = {cb_index};
    TensorAccessorArgs(*input_buffer).append_to(compile_time_args);

    KernelHandle kernel_id = CreateKernel(
        program,
        "path/to/kernel.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig{compile_time_args}
    );

    // 7. RUNTIME ARGS
    SetRuntimeArgs(program, kernel_id, core, {input_buffer->address()});

    // 8. PROGRAM EXECUTION
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // 9. READ RESULTS
    std::vector<bfloat16> result_vec(constants::TILE_HW);
    distributed::EnqueueReadMeshBuffer(cq, result_vec, output_buffer, true);

    // Untilize results
    result_vec = untilize_nfaces(result_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // 10. CLEANUP
    mesh_device->close();

    return 0;
}
```

KEY POINTS:
- Use `<tt-metalium/header.hpp>` NOT "tt_metal/header.hpp"
- Use `distributed::MeshDevice` NOT `Device`
- Use `distributed::MeshBuffer::create()` NOT `Buffer::create()`
- Use `distributed::EnqueueWriteMeshBuffer()` NOT `EnqueueWriteBuffer()`
- All distributed APIs are in the `distributed::` namespace



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


### Example: /home/m48chen/tt-metal/tt_metal/programming_examples/tests/mesh_device_test.cpp


```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/system_mesh.hpp>

using namespace tt::tt_metal;

int main() {
    // Make sure creating a mesh device with the full system mesh shape works. This is specifically meant to catch
    // regressions such as https://github.com/tenstorrent/tt-metal/issues/30899. Ideally it should be ran against
    // a single host context (requires building `--without-distributed`) and the MPI one.
    auto mesh_shape = distributed::SystemMesh::instance().shape();
    auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create(mesh_device_config);

    // Close the device
    if (!mesh_device->close()) {
        return 1;
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

- Follow the canonical template structure exactly

- CMakeLists.txt MUST include: find_package(TT-Metalium) and target_link_libraries(...TT::Metalium)
```

### User
```
Generate complete host code (.cpp file) AND CMakeLists.txt for diode current equation (I = isat × (exp(V/vj) - 1)) (single-core).

Requirements:
- Follow the canonical template structure EXACTLY
- Use correct headers: `#include <tt-metalium/host_api.hpp>` with angle brackets
- Use `distributed::MeshDevice::create_unit_mesh()` for device setup
- Create DRAM buffers using `distributed::MeshBuffer::create()`
- Configure CB_0 (V), CB_1 (vj), CB_2 (isat constant), CB_16 (output). Initialize CB_2 with the constant isat value.
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


================================================================================
YOUR PREVIOUS OUTPUT (the code you generated):
================================================================================

## Compute Kernel
```cpp
// COMPUTE KERNEL: diode_equation.cpp
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"

namespace NAMESPACE {
void MAIN {
    // Compile time args
    constexpr uint32_t src_cb_index_V = 0;
    constexpr uint32_t src_cb_index_vj = 1;
    constexpr uint32_t src_cb_index_isat = 2;
    constexpr uint32_t result_cb_index = 16;

    // Constants
    constexpr uint32_t one_tile = 1;

    // Initialize the SFPU
    init_sfpu(src_cb_index_V, result_cb_index);

    // Wait for the SFPU to have registers available for us to use during
    // the computation.
    tile_regs_acquire();

    // Wait for data to show up in the circular buffer and copy it from
    // the circular buffer to registers so the SFPU can use it.
    cb_wait_front(src_cb_index_V, one_tile);
    cb_wait_front(src_cb_index_vj, one_tile);
    cb_wait_front(src_cb_index_isat, one_tile);
    copy_tile(src_cb_index_V, /*offset*/ 0, /*register_offset*/ 0);
    copy_tile(src_cb_index_vj, /*offset*/ 0, /*register_offset*/ 1);
    copy_tile(src_cb_index_isat, /*offset*/ 0, /*register_offset*/ 2);

    //
    // Fused operations
    //
    // Compute the division of V by vj using the SFPU.
    div_binary_tile_init();
    div_binary_tile(0, 1, 0);  // V/vj

    // Compute the exponentiation of the result using the SFPU.
    exp_tile_init();
    exp_tile(0);  // exp(V/vj)

    // Subtract 1 from the result using the SFPU.
    sub_unary_tile_init();
    sub_unary_tile(0, 1);  // exp(V/vj) - 1

    // Multiply the result by isat using the SFPU.
    mul_binary_tile_init();
    mul_binary_tile(0, 2, 0);  // isat * (exp(V/vj) - 1)

    // Wait for result to be done and data stored back to the circular buffer
    tile_regs_commit();
    tile_regs_wait();

    // Reserve output tile
    cb_reserve_back(result_cb_index, one_tile);

    pack_tile(0, result_cb_index);  // copy tile 0 from the registers to the CB

    // We don't need the input tile anymore, mark it as consumed
    cb_pop_front(src_cb_index_V, one_tile);
    cb_pop_front(src_cb_index_vj, one_tile);
    cb_pop_front(src_cb_index_isat, one_tile);

    // Done with the registers, we can release them for the next SFPU operation
    tile_regs_release();

    // Mark the tile as ready for the writer kernel to write to DRAM
    cb_push_back(result_cb_index, one_tile);
}
}  // namespace NAMESPACE
```

## Reader Kernel
```cpp
// READER KERNEL: reader_binary_1_tile.cpp
#include "dataflow_api.h"
#include "tt-metalium/constants.hpp"

#include <cstdint>
#include <cstring>

void kernel_main() {
    // Runtime args
    const uint32_t input_buffer_addr_V = get_arg_val<uint32_t>(0);
    const uint32_t input_buffer_addr_vj = get_arg_val<uint32_t>(1);
    const uint32_t input_buffer_addr_isat = get_arg_val<uint32_t>(2);

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
    const auto interleaved_accessor_isat =
        TensorAccessor(interleaved_accessor_args, input_buffer_addr_isat, input_data_tile_size_bytes);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Read input value data
    cb_reserve_back(src_cb_index_V, one_tile);
    const uint32_t l1_write_addr_V = get_write_ptr(src_cb_index_V);
    noc_async_read_tile(0, interleaved_accessor_V, l1_write_addr_V);
    cb_push_back(src_cb_index_V, one_tile);

    cb_reserve_back(src_cb_index_vj, one_tile);
    const uint32_t l1_write_addr_vj = get_write_ptr(src_cb_index_vj);
    noc_async_read_tile(0, interleaved_accessor_vj, l1_write_addr_vj);
    cb_push_back(src_cb_index_vj, one_tile);

    cb_reserve_back(src_cb_index_isat, one_tile);
    const uint32_t l1_write_addr_isat = get_write_ptr(src_cb_index_isat);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_isat);
    for (uint32_t i = 0; i < tt::constants::TILE_HW; i++) {
        ptr[i] = 1.0f;  // Initialize isat with a constant value
    }
    cb_push_back(src_cb_index_isat, one_tile);

    noc_async_read_barrier();
}
```

## Writer Kernel
```cpp
// WRITER KERNEL: writer_1_tile.cpp
#include "dataflow_api.h"

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t output_buffer_addr = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr uint32_t result_cb_index = 16;

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

## Host Code
```cpp
// HOST CODE: diode_equation_single_v2.cpp
// SPDX-FileCopyrightText: 2025 Tenstorrent AI ULC
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

using namespace tt;
using namespace tt::tt_metal;

inline float check_bfloat16_vector_pcc(const std::vector<bfloat16>& vec_a, const std::vector<bfloat16>& vec_b) {
    float x_mean = 0.0f;
    float y_mean = 0.0f;

    for (size_t i = 0; i < vec_a.size(); i++) {
        x_mean += static_cast<float>(vec_a[i]);
        y_mean += static_cast<float>(vec_b[i]);
    }

    x_mean /= vec_a.size();
    y_mean /= vec_b.size();

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

    float correlation_coefficient_ = covariance / (std::sqrt(x_stddev) * std::sqrt(y_stddev));
    return correlation_coefficient_;
}

int main() {
    // 1. DEVICE SETUP - use distributed::MeshDevice for modern TT-Metal
    std::shared_ptr<distributed::MeshDevice> mesh_device =
        distributed::MeshDevice::create_unit_mesh(0);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range =
        distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    // 2. CORE RANGE SETUP
    constexpr CoreCoord core = {0, 0};

    // 3. INPUT DATA PREPARATION
    std::vector<bfloat16> input_vec(constants::TILE_HW);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.f, 1.0f);
    for (bfloat16& v : input_vec) {
        v = bfloat16(dist(rng));
    }

    // Tilize input data for device
    input_vec = tilize_nfaces(input_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // 4. DRAM BUFFER CREATION - use distributed::MeshBuffer
    constexpr uint32_t single_tile_size =
        sizeof(bfloat16) * constants::TILE_HEIGHT * constants::TILE_WIDTH;
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM
    };
    distributed::ReplicatedBufferConfig buffer_config{
        .size = sizeof(bfloat16) * input_vec.size()
    };
    std::shared_ptr<distributed::MeshBuffer> input_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> output_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    // Write input data to device
    distributed::EnqueueWriteMeshBuffer(cq, input_buffer, input_vec, false);

    // 5. CIRCULAR BUFFER SETUP
    constexpr uint32_t cb_index_v = CBIndex::c_0;
    constexpr uint32_t cb_index_vj = CBIndex::c_1;
    constexpr uint32_t cb_index_isat = CBIndex::c_2;
    constexpr uint32_t cb_index_output = CBIndex::c_16;
    CircularBufferConfig cb_config_v =
        CircularBufferConfig(single_tile_size, {{cb_index_v, tt::DataFormat::Float16_b}})
            .set_page_size(cb_index_v, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_config_v);
    CircularBufferConfig cb_config_vj =
        CircularBufferConfig(single_tile_size, {{cb_index_vj, tt::DataFormat::Float16_b}})
            .set_page_size(cb_index_vj, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_config_vj);
    CircularBufferConfig cb_config_isat =
        CircularBufferConfig(single_tile_size, {{cb_index_isat, tt::DataFormat::Float16_b}})
            .set_page_size(cb_index_isat, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_config_isat);
    CircularBufferConfig cb_config_output =
        CircularBufferConfig(single_tile_size, {{cb_index_output, tt::DataFormat::Float16_b}})
            .set_page_size(cb_index_output, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_config_output);

    // Initialize isat constant
    std::vector<bfloat16> isat_vec(constants::TILE_HW, bfloat16(0.1f));
    isat_vec = tilize_nfaces(isat_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);
    distributed::EnqueueWriteMeshBuffer(cq, input_buffer, isat_vec, false);

    // 6. KERNEL CREATION
    std::vector<uint32_t> compile_time_args = {cb_index_v, cb_index_vj, cb_index_isat, cb_index_output};
    TensorAccessorArgs(*input_buffer).append_to(compile_time_args);
    KernelHandle kernel_id_reader = CreateKernel(
        program,
        "path/to/kernel_reader.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig{compile_time_args}
    );
    KernelHandle kernel_id_compute = CreateKernel(
        program,
        "path/to/kernel_compute.cpp",
        core,
        tt::tt_metal::ComputeConfig{.compile_args = compile_time_args}
    );
    KernelHandle kernel_id_writer = CreateKernel(
        program,
        "path/to/kernel_writer.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig{compile_time_args}
    );

    // 7. RUNTIME ARGS
    SetRuntimeArgs(program, kernel_id_reader, core, {input_buffer->address()});
    SetRuntimeArgs(program, kernel_id_compute, core, {input_buffer->address()});
    SetRuntimeArgs(program, kernel_id_writer, core, {output_buffer->address()});

    // 8. PROGRAM EXECUTION
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // 9. READ RESULTS
    std::vector<bfloat16> result_vec(constants::TILE_HW);
    distributed::EnqueueReadMeshBuffer(cq, result_vec, output_buffer, true);

    // Untilize results
    result_vec = untilize_nfaces(result_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // 10. CPU GOLDEN VALIDATION
    std::vector<bfloat16> golden_vec(constants::TILE_HW);
    for (size_t i = 0; i < input_vec.size(); ++i) {
        float v = static_cast<float>(input_vec[i]);
        float vj = 0.02585f; // thermal voltage
        float isat = 0.1f;
        golden_vec[i] = bfloat16(isat * (std::exp(v / vj) - 1));
    }
    float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
    fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
    TT_FATAL(pearson > 0.999, "PCC not high enough. Result PCC: {}, Expected PCC: 0.999", pearson);

    // 11. CLEANUP
    mesh_device->close();

    return 0;
}
```

## CMakeLists.txt
```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.22...3.30)
project(diode_equation_single_v2)

add_executable(diode_equation_single_v2)
target_sources(diode_equation_single_v2 PRIVATE diode_equation_single_v2.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(diode_equation_single_v2 PUBLIC TT::Metalium)
```

================================================================================
COMPILATION ERRORS (what went wrong):
================================================================================
```

COMPILATION ERRORS (6 errors):
  - profiler.cpp:1632: no member named 'to_string' in 'tracy::TTDeviceMarker'
  - profiler.cpp:1640: no member named 'to_string' in 'tracy::TTDeviceMarker'
  - profiler.cpp:1641: no member named 'to_string' in 'tracy::TTDeviceMarker'
  - profiler.cpp:1653: no member named 'to_string' in 'tracy::TTDeviceMarker'
  - profiler.cpp:1654: no member named 'to_string' in 'tracy::TTDeviceMarker'
```

```

## User
```
The code you generated in the previous iteration failed to compile with the errors shown above.

Please fix ALL the errors and provide the complete, corrected code.

Output format - provide ALL files with fixes applied:

```cpp
// COMPUTE KERNEL
[complete corrected compute kernel code]
```

```cpp
// READER KERNEL
[complete corrected reader kernel code]
```

```cpp
// WRITER KERNEL
[complete corrected writer kernel code]
```

```cpp
// HOST CODE
[complete corrected host code]
```

```cmake
# CMakeLists.txt
[complete corrected cmake file]
```

CRITICAL: Provide COMPLETE files, not just the changed sections.
```


## LLM Response
```
## Compute Kernel
```cpp
// COMPUTE KERNEL: diode_equation.cpp
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"

namespace NAMESPACE {
void MAIN {
    // Compile time args
    constexpr uint32_t src_cb_index_V = 0;
    constexpr uint32_t src_cb_index_vj = 1;
    constexpr uint32_t src_cb_index_isat = 2;
    constexpr uint32_t result_cb_index = 16;

    // Constants
    constexpr uint32_t one_tile = 1;

    // Initialize the SFPU
    init_sfpu(src_cb_index_V, result_cb_index);

    // Wait for the SFPU to have registers available for us to use during
    // the computation.
    tile_regs_acquire();

    // Wait for data to show up in the circular buffer and copy it from
    // the circular buffer to registers so the SFPU can use it.
    cb_wait_front(src_cb_index_V, one_tile);
    cb_wait_front(src_cb_index_vj, one_tile);
    cb_wait_front(src_cb_index_isat, one_tile);
    copy_tile(src_cb_index_V, /*offset*/ 0, /*register_offset*/ 0);
    copy_tile(src_cb_index_vj, /*offset*/ 0, /*register_offset*/ 1);
    copy_tile(src_cb_index_isat, /*offset*/ 0, /*register_offset*/ 2);

    //
    // Fused operations
    //
    // Compute the division of V by vj using the SFPU.
    div_binary_tile_init();
    div_binary_tile(0, 1, 0);  // V/vj

    // Compute the exponentiation of the result using the SFPU.
    exp_tile_init();
    exp_tile(0);  // exp(V/vj)

    // Subtract 1 from the result using the SFPU.
    sub_unary_tile_init();
    sub_unary_tile(0, 1);  // exp(V/vj) - 1

    // Multiply the result by isat using the SFPU.
    mul_binary_tile_init();
    mul_binary_tile(0, 2, 0);  // isat * (exp(V/vj) - 1)

    // Wait for result to be done and data stored back to the circular buffer
    tile_regs_commit();
    tile_regs_wait();

    // Reserve output tile
    cb_reserve_back(result_cb_index, one_tile);

    pack_tile(0, result_cb_index);  // copy tile 0 from the registers to the CB

    // We don't need the input tile anymore, mark it as consumed
    cb_pop_front(src_cb_index_V, one_tile);
    cb_pop_front(src_cb_index_vj, one_tile);
    cb_pop_front(src_cb_index_isat, one_tile);

    // Done with the registers, we can release them for the next SFPU operation
    tile_regs_release();

    // Mark the tile as ready for the writer kernel to write to DRAM
    cb_push_back(result_cb_index, one_tile);
}
}  // namespace NAMESPACE
```

## Reader Kernel
```cpp
// READER KERNEL: reader_binary_1_tile.cpp
#include "dataflow_api.h"
#include "tt-metalium/constants.hpp"

#include <cstdint>
#include <cstring>

void kernel_main() {
    // Runtime args
    const uint32_t input_buffer_addr_V = get_arg_val<uint32_t>(0);
    const uint32_t input_buffer_addr_vj = get_arg_val<uint32_t>(1);
    const uint32_t input_buffer_addr_isat = get_arg_val<uint32_t>(2);

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
    const auto interleaved_accessor_isat =
        TensorAccessor(interleaved_accessor_args, input_buffer_addr_isat, input_data_tile_size_bytes);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Read input value data
    cb_reserve_back(src_cb_index_V, one_tile);
    const uint32_t l1_write_addr_V = get_write_ptr(src_cb_index_V);
    noc_async_read_tile(0, interleaved_accessor_V, l1_write_addr_V);
    cb_push_back(src_cb_index_V, one_tile);

    cb_reserve_back(src_cb_index_vj, one_tile);
    const uint32_t l1_write_addr_vj = get_write_ptr(src_cb_index_vj);
    noc_async_read_tile(0, interleaved_accessor_vj, l1_write_addr_vj);
    cb_push_back(src_cb_index_vj, one_tile);

    cb_reserve_back(src_cb_index_isat, one_tile);
    const uint32_t l1_write_addr_isat = get_write_ptr(src_cb_index_isat);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_isat);
    for (uint32_t i = 0; i < tt::constants::TILE_HW; i++) {
        ptr[i] = 1.0f;  // Initialize isat with a constant value
    }
    cb_push_back(src_cb_index_isat, one_tile);

    noc_async_read_barrier();
}
```

## Writer Kernel
```cpp
// WRITER KERNEL: writer_1_tile.cpp
#include "dataflow_api.h"

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t output_buffer_addr = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr uint32_t result_cb_index = 16;

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

## Host Code
```cpp
// HOST CODE: diode_equation_single_v2.cpp
// SPDX-FileCopyrightText: 2025 Tenstorrent AI ULC
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

using namespace tt;
using namespace tt::tt_metal;

inline float check_bfloat16_vector_pcc(const std::vector<bfloat16>& vec_a, const std::vector<bfloat16>& vec_b) {
    float x_mean = 0.0f;
    float y_mean = 0.0f;

    for (size_t i = 0; i < vec_a.size(); i++) {
        x_mean += static_cast<float>(vec_a[i]);
        y_mean += static_cast<float>(vec_b[i]);
    }

    x_mean /= vec_a.size();
    y_mean /= vec_b.size();

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

    float correlation_coefficient_ = covariance / (std::sqrt(x_stddev) * std::sqrt(y_stddev));
    return correlation_coefficient_;
}

int main() {
    // 1. DEVICE SETUP - use distributed::MeshDevice for modern TT-Metal
    std::shared_ptr<distributed::MeshDevice> mesh_device =
        distributed::MeshDevice::create_unit_mesh(0);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range =
        distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    // 2. CORE RANGE SETUP
    constexpr CoreCoord core = {0, 0};

    // 3. INPUT DATA PREPARATION
    std::vector<bfloat16> input_vec(constants::TILE_HW);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.f, 1.0f);
    for (bfloat16& v : input_vec) {
        v = bfloat16(dist(rng));
    }

    // Tilize input data for device
    input_vec = tilize_nfaces(input_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // 4. DRAM BUFFER CREATION - use distributed::MeshBuffer
    constexpr uint32_t single_tile_size =
        sizeof(bfloat16) * constants::TILE_HEIGHT * constants::TILE_WIDTH;
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM
    };
    distributed::ReplicatedBufferConfig buffer_config{
        .size = sizeof(bfloat16) * input_vec.size()
    };
    std::shared_ptr<distributed::MeshBuffer> input_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> output_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    // Write input data to device
    distributed::EnqueueWriteMeshBuffer(cq, input_buffer, input_vec, false);

    // 5. CIRCULAR BUFFER SETUP
    constexpr uint32_t cb_index_v = CBIndex::c_0;
    constexpr uint32_t cb_index_vj = CBIndex::c_1;
    constexpr uint32_t cb_index_isat = CBIndex::c_2;
    constexpr uint32_t cb_index_output = CBIndex::c_16;
    CircularBufferConfig cb_config_v =
        CircularBufferConfig(single_tile_size, {{cb_index_v, tt::DataFormat::Float16_b}})
            .set_page_size(cb_index_v, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_config_v);
    CircularBufferConfig cb_config_vj =
        CircularBufferConfig(single_tile_size, {{cb_index_vj, tt::DataFormat::Float16_b}})
            .set_page_size(cb_index_vj, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_config_vj);
    CircularBufferConfig cb_config_isat =
        CircularBufferConfig(single_tile_size, {{cb_index_isat, tt::DataFormat::Float16_b}})
            .set_page_size(cb_index_isat, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_config_isat);
    CircularBufferConfig cb_config_output =
        CircularBufferConfig(single_tile_size, {{cb_index_output, tt::DataFormat::Float16_b}})
            .set_page_size(cb_index_output, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_config_output);

    // Initialize isat constant
    std::vector<bfloat16> isat_vec(constants::TILE_HW, bfloat16(0.1f));
    isat_vec = tilize_nfaces(isat_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);
    distributed::EnqueueWriteMeshBuffer(cq, input_buffer, isat_vec, false);

    // 6. KERNEL CREATION
    std::vector<uint32_t> compile_time_args = {cb_index_v, cb_index_vj, cb_index_isat, cb_index_output};
    TensorAccessorArgs(*input_buffer).append_to(compile_time_args);
    KernelHandle kernel_id_reader = CreateKernel(
        program,
        "path/to/kernel_reader.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig{compile_time_args}
    );
    KernelHandle kernel_id_compute = CreateKernel(
        program,
        "path/to/kernel_compute.cpp",
        core,
        tt::tt_metal::ComputeConfig{.compile_args = compile_time_args}
    );
    KernelHandle kernel_id_writer = CreateKernel(
        program,
        "path/to/kernel_writer.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig{compile_time_args}
    );

    // 7. RUNTIME ARGS
    SetRuntimeArgs(program, kernel_id_reader, core, {input_buffer->address()});
    SetRuntimeArgs(program, kernel_id_compute, core, {input_buffer->address()});
    SetRuntimeArgs(program, kernel_id_writer, core, {output_buffer->address()});

    // 8. PROGRAM EXECUTION
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // 9. READ RESULTS
    std::vector<bfloat16> result_vec(constants::TILE_HW);
    distributed::EnqueueReadMeshBuffer(cq, result_vec, output_buffer, true);

    // Untilize results
    result_vec = untilize_nfaces(result_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // 10. CPU GOLDEN VALIDATION
    std::vector<bfloat16> golden_vec(constants::TILE_HW);
    for (size_t i = 0; i < input_vec.size(); ++i) {
        float v = static_cast<float>(input_vec[i]);
        float vj = 0.02585f; // thermal voltage
        float isat = 0.1f;
        golden_vec[i] = bfloat16(isat * (std::exp(v / vj) - 1));
    }
    float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
    fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
    TT_FATAL(pearson > 0.999, "PCC not high enough. Result PCC: {}, Expected PCC: 0.999", pearson);

    // 11. CLEANUP
    mesh_device->close();

    return 0;
}
```

## CMakeLists.txt
```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.22...3.30)
project(diode_equation_single_v2)

add_executable(diode_equation_single_v2)
target_sources(diode_equation_single_v2 PRIVATE diode_equation_single_v2.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(diode_equation_single_v2 PUBLIC TT::Metalium)
```
```
