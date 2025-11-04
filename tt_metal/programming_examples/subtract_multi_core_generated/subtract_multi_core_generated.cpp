// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "tt-metalium/constants.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    // Ensure printing from kernel is enabled
    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        fmt::print(
            "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of the Data "
            "Movement kernels.\n");
        fmt::print("WARNING: For example, export TT_METAL_DPRINT_CORES=0,0\n");
    }

    fmt::print("Multi-core Element-wise Subtraction Example (C = A - B)\n");
    fmt::print("========================================================\n");

    constexpr uint32_t M = 256;                // Height in elements
    constexpr uint32_t N = 512;                // Width in elements
    constexpr uint32_t Mt = M / TILE_HEIGHT;   // Number of tiles in M dimension (8 tiles)
    constexpr uint32_t Nt = N / TILE_WIDTH;    // Number of tiles in N dimension (16 tiles)
    constexpr uint32_t total_tiles = Mt * Nt;  // Total tiles = 128 tiles

    fmt::print("Matrix dimensions: {}x{} elements\n", M, N);
    fmt::print("Tile dimensions: {}x{} tiles ({} total tiles)\n", Mt, Nt, total_tiles);

    // Create a UnitMesh (1x1 MeshDevice)
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);

    // Set up mesh command queue, workload, device range, and program for multi-core execution
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program{};

    // Get the compute grid size to determine how many cores are available
    auto compute_grid = mesh_device->compute_with_storage_grid_size();
    fmt::print("Available compute grid: {}x{} cores\n", compute_grid.x, compute_grid.y);

    // Use the split_work_to_cores utility function to distribute work across available cores
    // This function optimally distributes tiles across cores for load balancing
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] =
        split_work_to_cores(compute_grid, total_tiles);

    fmt::print("Work distribution: {} cores total\n", num_cores);
    fmt::print("Group 1: {} cores with {} tiles each\n", core_group_1.num_cores(), work_per_core1);
    if (core_group_2.num_cores() > 0) {
        fmt::print("Group 2: {} cores with {} tiles each\n", core_group_2.num_cores(), work_per_core2);
    }

    // Buffer configuration
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;  // 2048 bytes per tile
    const uint32_t total_data_size = single_tile_size * total_tiles;

    // Create DRAM buffers for input and output (replicated per device across the mesh)
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = total_data_size};

    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    fmt::print("Work distribution: {} tiles across {} cores\n", total_tiles, num_cores);

    // Create circular buffers with appropriate configuration
    auto make_cb_config = [](CBIndex cb_index) {
        return CircularBufferConfig(1 * 32 * 32 * sizeof(bfloat16), {{cb_index, DataFormat::Float16_b}})
            .set_page_size(cb_index, 32 * 32 * sizeof(bfloat16));
    };

    CreateCircularBuffer(program, all_cores, make_cb_config(CBIndex::c_0));
    CreateCircularBuffer(program, all_cores, make_cb_config(CBIndex::c_1));
    CreateCircularBuffer(program, all_cores, make_cb_config(CBIndex::c_16));

    // Generate compile-time arguments for TensorAccessorArgs (writer kernel only - reader uses InterleavedAddrGenFast)
    std::vector<uint32_t> writer_compile_time_args = {};

    // Add TensorAccessorArgs for output tensor (writer needs dst)
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);

    // Create kernels using our generated multi-core kernel files
    auto reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "subtract_multi_core_generated/kernels/dataflow/reader_binary_tiles_partitioned.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "subtract_multi_core_generated/kernels/dataflow/writer_tiles_partitioned.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    auto compute_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "subtract_multi_core_generated/kernels/compute/subtract_2_tiles.cpp",
        all_cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .math_approx_mode = false});

    // Create input data with larger dataset
    std::vector<bfloat16> src0_vec(M * N);
    std::vector<bfloat16> src1_vec(M * N);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist1(5.0f, 15.0f);  // Higher range for A
    std::uniform_real_distribution<float> dist2(0.0f, 5.0f);   // Lower range for B (so A - B > 0)

    fmt::print("Generating random input data...\n");
    for (uint32_t i = 0; i < M * N; ++i) {
        src0_vec[i] = bfloat16(dist1(rng));  // A: values 5-15
        src1_vec[i] = bfloat16(dist2(rng));  // B: values 0-5
    }

    // Upload data to device
    fmt::print("Uploading data to device...\n");
    EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, false);
    EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1_vec, false);

    // Set Runtime Arguments for Kernels using the same pattern as matmul
    // Each core needs to know which portion of the work it's responsible for
    uint32_t work_offset = 0;
    auto work_groups = {std::make_pair(core_group_1, work_per_core1), std::make_pair(core_group_2, work_per_core2)};

    fmt::print("Setting runtime arguments for kernels...\n");
    // Iterate through each work group and assign work to cores
    for (const auto& [ranges, work_per_core] : work_groups) {
        for (const auto& range : ranges.ranges()) {
            for (const auto& core : range) {
                // Reader kernel args: src0_addr, src1_addr, num_tiles, start_tile_id
                SetRuntimeArgs(
                    program,
                    reader_kernel_id,
                    core,
                    {(uint32_t)src0_dram_buffer->address(),
                     (uint32_t)src1_dram_buffer->address(),
                     work_per_core,
                     work_offset});

                // Compute kernel args: num_output_tiles
                SetRuntimeArgs(program, compute_kernel_id, core, {work_per_core});

                // Writer kernel args: dst_addr, num_tiles, start_tile_id
                SetRuntimeArgs(
                    program,
                    writer_kernel_id,
                    core,
                    {(uint32_t)dst_dram_buffer->address(), work_per_core, work_offset});

                work_offset += work_per_core;
            }
        }
    }

    // Execute the program
    fmt::print("Executing multi-core subtraction...\n");
    auto start_time = std::chrono::high_resolution_clock::now();

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    fmt::print("Execution completed in {} microseconds\n", duration.count());

    // Read results
    fmt::print("Reading results from device...\n");
    std::vector<bfloat16> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

    // Verify results
    fmt::print("Verifying results...\n");
    uint32_t num_errors = 0;
    const float tolerance = 3e-1f;

    for (uint32_t i = 0; i < M * N; ++i) {
        float expected = float(src0_vec[i]) - float(src1_vec[i]);
        float actual = float(result_vec[i]);
        float error = std::abs(actual - expected);

        if (error > tolerance) {
            if (num_errors < 10) {  // Only print first 10 errors
                fmt::print(
                    "Error at index {}: expected {:.6f}, got {:.6f}, error {:.6f}\n", i, expected, actual, error);
            }
            num_errors++;
        }
    }

    // Print summary
    fmt::print("\n=== RESULTS SUMMARY ===\n");
    fmt::print("Matrix dimensions: {}x{} elements\n", M, N);
    fmt::print("Total elements: {}\n", M * N);
    fmt::print("Cores used: {} out of {} available\n", num_cores, compute_grid.x * compute_grid.y);
    fmt::print("Execution time: {} microseconds\n", duration.count());

    if (num_errors == 0) {
        fmt::print("✅ SUCCESS: All {} elements computed correctly!\n", M * N);
        fmt::print("Multi-core subtraction (C = A - B) completed successfully.\n");
    } else {
        fmt::print("❌ FAILED: {} out of {} elements had errors > {:.6f}\n", num_errors, M * N, tolerance);
        fmt::print("Accuracy: {:.2f}%\n", 100.0f * (M * N - num_errors) / (M * N));
    }

    // Show some sample results
    fmt::print("\nSample results (first 8 elements):\n");
    for (uint32_t i = 0; i < std::min(8u, M * N); ++i) {
        fmt::print(
            "  A[{}]={:.3f} - B[{}]={:.3f} = C[{}]={:.3f}\n",
            i,
            float(src0_vec[i]),
            i,
            float(src1_vec[i]),
            i,
            float(result_vec[i]));
    }

    return num_errors == 0 ? 0 : 1;
}
