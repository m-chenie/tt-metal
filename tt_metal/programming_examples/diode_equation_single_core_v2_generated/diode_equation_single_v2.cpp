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

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// Helper function for PCC calculation
float check_bfloat16_vector_pcc(const std::vector<bfloat16>& vec_a, const std::vector<bfloat16>& vec_b) {
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
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    // 2. CORE RANGE SETUP
    constexpr CoreCoord core = {0, 0};

    // 3. INPUT DATA PREPARATION
    std::vector<bfloat16> input_vec(constants::TILE_HW);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : input_vec) {
        v = bfloat16(dist(rng));
    }

    // Tilize input data for device
    input_vec = tilize_nfaces(input_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // 4. DRAM BUFFER CREATION - use distributed::MeshBuffer
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * constants::TILE_HEIGHT * constants::TILE_WIDTH;
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = sizeof(bfloat16) * input_vec.size()};
    std::shared_ptr<distributed::MeshBuffer> input_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> output_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    // Write input data to device
    distributed::EnqueueWriteMeshBuffer(cq, input_buffer, input_vec, false);

    // 5. CIRCULAR BUFFER SETUP
    constexpr uint32_t cb_index_input0 = CBIndex::c_0;
    constexpr uint32_t cb_index_input1 = CBIndex::c_1;
    constexpr uint32_t cb_index_output = CBIndex::c_16;
    CircularBufferConfig cb_config_input0 =
        CircularBufferConfig(single_tile_size, {{cb_index_input0, tt::DataFormat::Float16_b}})
            .set_page_size(cb_index_input0, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_config_input0);
    CircularBufferConfig cb_config_input1 =
        CircularBufferConfig(single_tile_size, {{cb_index_input1, tt::DataFormat::Float16_b}})
            .set_page_size(cb_index_input1, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_config_input1);
    CircularBufferConfig cb_config_output =
        CircularBufferConfig(single_tile_size, {{cb_index_output, tt::DataFormat::Float16_b}})
            .set_page_size(cb_index_output, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_config_output);

    // 6. KERNEL CREATION
    std::vector<uint32_t> compile_time_args = {cb_index_input0, cb_index_input1};
    TensorAccessorArgs(*input_buffer).append_to(compile_time_args);
    KernelHandle kernel_id_reader = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "diode_equation_single_core_v2_generated/kernels/dataflow/reader_binary_1_tile.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig{compile_time_args});
    compile_time_args = {cb_index_input0, cb_index_input1, cb_index_output};
    KernelHandle kernel_id_compute = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "diode_equation_single_core_v2_generated/kernels/compute/diode_equation.cpp",
        core,
        tt::tt_metal::ComputeConfig{.compile_args = compile_time_args});
    compile_time_args = {cb_index_output};
    TensorAccessorArgs(*output_buffer).append_to(compile_time_args);
    KernelHandle kernel_id_writer = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "diode_equation_single_core_v2_generated/kernels/dataflow/writer_binary_1_tile.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig{compile_time_args});

    // 7. RUNTIME ARGS
    SetRuntimeArgs(program, kernel_id_reader, core, {input_buffer->address()});
    SetRuntimeArgs(program, kernel_id_compute, core, {input_buffer->address(), output_buffer->address()});
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

    // CPU Golden Validation
    std::vector<bfloat16> golden_vec(result_vec.size());
    for (size_t i = 0; i < input_vec.size(); ++i) {
        float v = static_cast<float>(input_vec[i]);
        float isat = 1.0f;  // Assuming isat = 1.0 for simplicity
        float vj = 1.0f;    // Assuming vj = 1.0 for simplicity
        golden_vec[i] = bfloat16(isat * (std::exp(v / vj) - 1.0f));
    }

    // PCC Check
    float pcc = check_bfloat16_vector_pcc(golden_vec, result_vec);
    fmt::print("PCC: {}\n", pcc);
    TT_FATAL(pcc > 0.99, "PCC not high enough. Result PCC: {}, Expected PCC: 0.99", pcc);

    // 10. CLEANUP
    mesh_device->close();

    return 0;
}
