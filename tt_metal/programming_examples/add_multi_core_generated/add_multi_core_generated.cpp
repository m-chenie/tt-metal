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

    fmt::print("Multi-core Element-wise Addition Example\n");
    fmt::print("========================================\n");

    // Problem size configuration - make it bigger like matmul
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

    fmt::print("Created DRAM buffers: {} bytes each\n", total_data_size);

    // Create circular buffers for all cores with optimized size for streaming
    constexpr uint32_t num_tiles_cb = 4;  // Buffer more tiles for better throughput
    auto make_cb_config = [&](CBIndex cb_index) {
        return CircularBufferConfig(num_tiles_cb * single_tile_size, {{cb_index, DataFormat::Float16_b}})
            .set_page_size(cb_index, single_tile_size);
    };

    CreateCircularBuffer(program, all_cores, make_cb_config(CBIndex::c_0));
    CreateCircularBuffer(program, all_cores, make_cb_config(CBIndex::c_1));
    CreateCircularBuffer(program, all_cores, make_cb_config(CBIndex::c_16));

    // Generate compile-time arguments for TensorAccessorArgs
    std::vector<uint32_t> reader_compile_time_args = {};
    std::vector<uint32_t> writer_compile_time_args = {};

    // Add TensorAccessorArgs for input tensors (reader needs both src0 and src1)
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);

    // Add TensorAccessorArgs for output tensor (writer needs dst)
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);

    // Create kernels using our generated multi-core kernel files with compile-time args
    auto reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "add_multi_core_generated/kernels/dataflow/reader_binary_tiles_partitioned.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    auto writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "add_multi_core_generated/kernels/dataflow/writer_tiles_partitioned.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    auto compute_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "add_multi_core_generated/kernels/compute/add_2_tiles.cpp",
        all_cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .math_approx_mode = false});

    // Create input data with larger dataset
    std::vector<bfloat16> src0_vec(M * N);
    std::vector<bfloat16> src1_vec(M * N);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist1(0.0f, 10.0f);
    std::uniform_real_distribution<float> dist2(0.0f, 5.0f);

    fmt::print("Generating random input data...\n");
    for (size_t i = 0; i < M * N; ++i) {
        src0_vec[i] = bfloat16(dist1(rng));
        src1_vec[i] = bfloat16(dist2(rng));
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
    fmt::print("Executing multi-core element-wise addition...\n");
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Read results
    fmt::print("Reading results from device...\n");
    std::vector<bfloat16> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

    // Verify results with sampling to avoid excessive output
    fmt::print("Verifying results...\n");
    bool success = true;
    uint32_t num_errors = 0;
    constexpr uint32_t max_errors_to_show = 5;

    for (size_t i = 0; i < M * N; ++i) {
        float expected = static_cast<float>(src0_vec[i]) + static_cast<float>(src1_vec[i]);
        float actual = static_cast<float>(result_vec[i]);
        if (std::abs(expected - actual) > 3e-1f) {
            if (num_errors < max_errors_to_show) {
                fmt::print(stderr, "Mismatch at index {}: expected {:.3f}, got {:.3f}\n", i, expected, actual);
            }
            num_errors++;
            success = false;
        }
    }

    if (!success) {
        if (num_errors > max_errors_to_show) {
            fmt::print(stderr, "... and {} more errors\n", num_errors - max_errors_to_show);
        }
        fmt::print(
            "Error: {} mismatches out of {} elements ({:.2f}% error rate)\n",
            num_errors,
            M * N,
            (100.0f * num_errors) / (M * N));
        mesh_device->close();
        return 1;
    } else {
        fmt::print("Success! Multi-core addition completed successfully!\n");
        fmt::print("Processed {}x{} elements ({} tiles) across {} cores\n", M, N, total_tiles, num_cores);
        fmt::print("Performance: {:.1f} elements per core on average\n", (float)(M * N) / num_cores);
    }

    mesh_device->close();
    return 0;
}
