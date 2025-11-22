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
    std::uniform_real_distribution<float> dist(0.f, 1.0f);
    for (bfloat16& v : input_vec) {
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
    constexpr uint32_t cb_index_v = CBIndex::c_0;
    constexpr uint32_t cb_index_vj = CBIndex::c_1;
    constexpr uint32_t cb_index_isat = CBIndex::c_2;
    constexpr uint32_t cb_index_output = CBIndex::c_16;
    CircularBufferConfig cb_config_v = CircularBufferConfig(single_tile_size, {{cb_index_v, tt::DataFormat::Float16_b}})
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
        program, "path/to/kernel_reader.cpp", core, tt::tt_metal::ReaderDataMovementConfig{compile_time_args});
    KernelHandle kernel_id_compute = CreateKernel(
        program, "path/to/kernel_compute.cpp", core, tt::tt_metal::ComputeConfig{.compile_args = compile_time_args});
    KernelHandle kernel_id_writer = CreateKernel(
        program, "path/to/kernel_writer.cpp", core, tt::tt_metal::WriterDataMovementConfig{compile_time_args});

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
        float vj = 0.02585f;  // thermal voltage
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
