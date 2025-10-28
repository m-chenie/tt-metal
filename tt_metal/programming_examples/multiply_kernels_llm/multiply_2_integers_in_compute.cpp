// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "tt-metalium/constants.hpp"
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t n_elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_WIDTH;
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * n_elements_per_tile;

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig distributed_buffer_config{.size = single_tile_size};

    auto src0_dram_buffer = distributed::MeshBuffer::create(distributed_buffer_config, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(distributed_buffer_config, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(distributed_buffer_config, dram_config, mesh_device.get());

    constexpr uint32_t num_tiles = 1;
    auto make_cb_config = [&](CBIndex cb_index) {
        return CircularBufferConfig(num_tiles * single_tile_size, {{cb_index, DataFormat::Float16_b}})
            .set_page_size(cb_index, single_tile_size);
    };

    tt_metal::CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_0));
    tt_metal::CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_1));
    tt_metal::CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_16));

    KernelHandle binary_reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "multiply_kernels_llm/kernels/dataflow/reader_binary_1_tile.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "multiply_kernels_llm/kernels/dataflow/writer_1_tile.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "multiply_kernels_llm/kernels/compute/multiply_2_tiles.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .math_approx_mode = false});

    std::vector<bfloat16> src0_vec(n_elements_per_tile);
    std::vector<bfloat16> src1_vec(n_elements_per_tile);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist1(2.0f, 5.0f);
    std::uniform_real_distribution<float> dist2(1.0f, 3.0f);
    for (size_t i = 0; i < n_elements_per_tile; ++i) {
        src0_vec[i] = bfloat16(dist1(rng));
        src1_vec[i] = bfloat16(dist2(rng));
    }

    EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, false);
    EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1_vec, false);

    SetRuntimeArgs(
        program,
        binary_reader_kernel_id,
        core,
        {(uint32_t)src0_dram_buffer->address(), (uint32_t)src1_dram_buffer->address()});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {});
    SetRuntimeArgs(program, unary_writer_kernel_id, core, {(uint32_t)dst_dram_buffer->address()});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<bfloat16> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

    bool success = true;
    int failures = 0;
    for (size_t i = 0; i < n_elements_per_tile; ++i) {
        float expected = static_cast<float>(src0_vec[i]) * static_cast<float>(src1_vec[i]);
        if (std::abs(expected - static_cast<float>(result_vec[i])) > 0.1f) {
            if (failures < 5) {
                fmt::print(
                    stderr,
                    "Mismatch at index {}: {} * {} = {} (expected {})\n",
                    i,
                    static_cast<float>(src0_vec[i]),
                    static_cast<float>(src1_vec[i]),
                    static_cast<float>(result_vec[i]),
                    expected);
            }
            failures++;
            success = false;
        }
    }
    if (!success) {
        fmt::print("Error: {} mismatches found!\n", failures);
    } else {
        fmt::print("Success: Multiplication kernel working correctly!\n");
        fmt::print(
            "  Sample: {} * {} = {}\n",
            static_cast<float>(src0_vec[0]),
            static_cast<float>(src1_vec[0]),
            static_cast<float>(result_vec[0]));
    }
    mesh_device->close();
    return success ? 0 : 1;
}
