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

int main() {
    // 1. DEVICE SETUP - use distributed::MeshDevice for modern TT-Metal
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    // 2. CORE RANGE SETUP
    constexpr CoreCoord core = {0, 0};

    // 3. INPUT DATA PREPARATION
    std::vector<bfloat16> input_vec_V(constants::TILE_HW);
    std::vector<bfloat16> input_vec_vj(constants::TILE_HW);
    std::vector<bfloat16> input_vec_isat(constants::TILE_HW);

    // Fill input vectors with data
    for (int i = 0; i < constants::TILE_HW; i++) {
        input_vec_V[i] = bfloat16(1.0f);
        input_vec_vj[i] = bfloat16(1.0f);
        input_vec_isat[i] = bfloat16(1.0f);
    }

    // Tilize input data for device
    input_vec_V = tilize_nfaces(input_vec_V, constants::TILE_WIDTH, constants::TILE_HEIGHT);
    input_vec_vj = tilize_nfaces(input_vec_vj, constants::TILE_WIDTH, constants::TILE_HEIGHT);
    input_vec_isat = tilize_nfaces(input_vec_isat, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // 4. DRAM BUFFER CREATION - use distributed::MeshBuffer
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * constants::TILE_HEIGHT * constants::TILE_WIDTH;
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config_V{.size = sizeof(bfloat16) * input_vec_V.size()};
    distributed::ReplicatedBufferConfig buffer_config_vj{.size = sizeof(bfloat16) * input_vec_vj.size()};
    distributed::ReplicatedBufferConfig buffer_config_isat{.size = sizeof(bfloat16) * input_vec_isat.size()};
    std::shared_ptr<distributed::MeshBuffer> input_buffer_V =
        distributed::MeshBuffer::create(buffer_config_V, dram_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> input_buffer_vj =
        distributed::MeshBuffer::create(buffer_config_vj, dram_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> input_buffer_isat =
        distributed::MeshBuffer::create(buffer_config_isat, dram_config, mesh_device.get());

    // Write input data to device
    distributed::EnqueueWriteMeshBuffer(cq, input_buffer_V, input_vec_V, false);
    distributed::EnqueueWriteMeshBuffer(cq, input_buffer_vj, input_vec_vj, false);
    distributed::EnqueueWriteMeshBuffer(cq, input_buffer_isat, input_vec_isat, false);

    // 5. CIRCULAR BUFFER SETUP
    constexpr uint32_t cb_index = CBIndex::c_0;
    CircularBufferConfig cb_config = CircularBufferConfig(single_tile_size, {{cb_index, tt::DataFormat::Float16_b}})
                                         .set_page_size(cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_config);

    // 6. KERNEL CREATION
    std::vector<uint32_t> compile_time_args = {cb_index};
    TensorAccessorArgs(*input_buffer_V).append_to(compile_time_args);
    TensorAccessorArgs(*input_buffer_vj).append_to(compile_time_args);
    TensorAccessorArgs(*input_buffer_isat).append_to(compile_time_args);

    KernelHandle kernel_id = CreateKernel(
        program, "path/to/diode_equation.cpp", core, tt::tt_metal::ReaderDataMovementConfig{compile_time_args});

    // 7. RUNTIME ARGS
    SetRuntimeArgs(
        program,
        kernel_id,
        core,
        {input_buffer_V->address(), input_buffer_vj->address(), input_buffer_isat->address()});

    // 8. PROGRAM EXECUTION
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // 9. READ RESULTS
    std::vector<bfloat16> result_vec(constants::TILE_HW);
    distributed::EnqueueReadMeshBuffer(cq, result_vec, input_buffer_V, true);

    // Untilize results
    result_vec = untilize_nfaces(result_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // 10. CLEANUP
    mesh_device->close();

    return 0;
}
