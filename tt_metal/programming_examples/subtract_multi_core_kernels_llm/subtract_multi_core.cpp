// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/device.hpp>
#include <fmt/core.h>

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// Reference implementation of element-wise subtraction.
// Array A and B are of the same size, and the output C is C = A - B.
void golden_subtract(
    std::vector<bfloat16>& a, std::vector<bfloat16>& b, std::vector<bfloat16>& output, uint32_t num_elements) {
    for (uint32_t i = 0; i < num_elements; i++) {
        float a_f = static_cast<float>(a[i]);
        float b_f = static_cast<float>(b[i]);
        output[i] = bfloat16(a_f - b_f);
    }
}

int main() {
    // Ensure printing from kernel is enabled
    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        fmt::print("WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to see kernel output.\n");
        fmt::print("WARNING: For example, export TT_METAL_DPRINT_CORES=0,0:0,1:1,0:1,1\n");
    }

    // Create a MeshDevice (single device for this example)
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    // Define problem size
    constexpr uint32_t Mt = 8;  // Height in tiles
    constexpr uint32_t Nt = 8;  // Width in tiles
    constexpr uint32_t num_tiles = Mt * Nt;
    constexpr uint32_t tile_size = TILE_HEIGHT * TILE_WIDTH;
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * tile_size;
    constexpr uint32_t total_elements = num_tiles * tile_size;

    fmt::print("Multi-core Element-wise Subtraction\n");
    fmt::print("Problem size: {}x{} tiles ({} elements)\n", Mt, Nt, total_elements);

    // Create program
    Program program = CreateProgram();

    // Get worker cores
    CoreCoord grid_size = mesh_device->get_device(0)->compute_with_storage_grid_size();

    fmt::print("Available compute grid size: {}x{}\n", grid_size.x, grid_size.y);

    // Distribute work across cores
    auto [num_cores_used, all_cores_set, core_range_1, core_range_2, units_per_core_1, units_per_core_2] =
        split_work_to_cores(CoreCoord{grid_size.x, grid_size.y}, num_tiles);

    // Use all cores that have work assigned
    CoreRangeSet core_range = all_cores_set;
    uint32_t cores_used = num_cores_used;

    fmt::print("Using {} cores for computation\n", cores_used);

    // Create DRAM buffers
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = num_tiles * single_tile_size};

    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    // Create circular buffers for each core
    constexpr uint32_t num_cb_tiles = 2;  // Double buffering
    auto make_cb_config = [&](CBIndex cb_index) {
        return CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb_index, DataFormat::Float16_b}})
            .set_page_size(cb_index, single_tile_size);
    };

    // Create circular buffers for all cores that will be used
    for (const auto& core_range_group : core_range.ranges()) {
        for (auto core : core_range_group) {
            CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_0));   // Input A
            CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_1));   // Input B
            CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_16));  // Output
        }
    }

    // Create kernels
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "subtract_multi_core_kernels_llm/kernels/dataflow/reader_binary_tiles_partitioned.cpp",
        core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "subtract_multi_core_kernels_llm/kernels/dataflow/writer_tiles_partitioned.cpp",
        core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    KernelHandle compute_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "subtract_multi_core_kernels_llm/kernels/compute/subtract_2_tiles.cpp",
        core_range,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .math_approx_mode = false});

    // Generate test data
    std::vector<bfloat16> src0_vec(total_elements);
    std::vector<bfloat16> src1_vec(total_elements);
    std::vector<bfloat16> expected_result(total_elements);

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist1(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist2(-5.0f, 5.0f);

    for (uint32_t i = 0; i < total_elements; i++) {
        src0_vec[i] = bfloat16(dist1(rng));
        src1_vec[i] = bfloat16(dist2(rng));
    }

    // Calculate expected results
    golden_subtract(src0_vec, src1_vec, expected_result, total_elements);

    // Upload data to device
    EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, false);
    EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1_vec, false);

    // Set runtime arguments for each core
    uint32_t start_tile_id = 0;

    // Set args for cores that do more work (core_range_1)
    for (const auto& core : core_range_1.ranges()) {
        for (auto core_coord : core) {
            // Reader kernel args: src0_addr, src1_addr, Mt, Kt, Nt, start_tile_id, num_tiles
            // For element-wise operations: Mt=height, Kt=1 (no K dimension), Nt=width
            SetRuntimeArgs(
                program,
                reader_kernel_id,
                core_coord,
                {
                    src0_dram_buffer->address(),
                    src1_dram_buffer->address(),
                    Mt,               // Mt - height in tiles
                    1,                // Kt - no K dimension for element-wise
                    Nt,               // Nt - width in tiles
                    start_tile_id,    // start_tile_id
                    units_per_core_1  // num_tiles
                });

            // Compute kernel args: num_tiles
            SetRuntimeArgs(program, compute_kernel_id, core_coord, {units_per_core_1});

            // Writer kernel args: dst_addr, start_tile_id, num_tiles
            SetRuntimeArgs(
                program, writer_kernel_id, core_coord, {dst_dram_buffer->address(), start_tile_id, units_per_core_1});

            start_tile_id += units_per_core_1;
        }
    }

    // Set args for cores that do less work (core_range_2, if any)
    if (!core_range_2.ranges().empty() && units_per_core_2 > 0) {
        for (const auto& core : core_range_2.ranges()) {
            for (auto core_coord : core) {
                SetRuntimeArgs(
                    program,
                    reader_kernel_id,
                    core_coord,
                    {
                        src0_dram_buffer->address(),
                        src1_dram_buffer->address(),
                        Mt,               // Mt - height in tiles
                        1,                // Kt - no K dimension for element-wise
                        Nt,               // Nt - width in tiles
                        start_tile_id,    // start_tile_id
                        units_per_core_2  // num_tiles
                    });

                // Compute kernel args: num_tiles
                SetRuntimeArgs(program, compute_kernel_id, core_coord, {units_per_core_2});

                // Writer kernel args: dst_addr, start_tile_id, num_tiles
                SetRuntimeArgs(
                    program,
                    writer_kernel_id,
                    core_coord,
                    {dst_dram_buffer->address(), start_tile_id, units_per_core_2});

                start_tile_id += units_per_core_2;
            }
        }
    }

    // Execute the program
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Read results
    std::vector<bfloat16> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

    // Verify results
    bool success = true;
    constexpr float tolerance = 3e-1f;  // tolerance for bfloat16 precision
    uint32_t num_errors = 0;
    constexpr uint32_t max_errors_to_print = 10;

    for (uint32_t i = 0; i < total_elements; i++) {
        float expected = static_cast<float>(expected_result[i]);
        float actual = static_cast<float>(result_vec[i]);
        float error = std::abs(expected - actual);

        if (error > tolerance) {
            if (num_errors < max_errors_to_print) {
                fmt::print(
                    stderr,
                    "Mismatch at index {}: expected {:.6f}, got {:.6f}, error {:.6f}\n",
                    i,
                    expected,
                    actual,
                    error);
            }
            num_errors++;
            success = false;
        }
    }

    if (!success) {
        fmt::print("Error: {} mismatches found out of {} elements!\n", num_errors, total_elements);
    } else {
        fmt::print("Success: All {} elements match expected values!\n", total_elements);
    }

    mesh_device->close();
    return success ? 0 : 1;
}
