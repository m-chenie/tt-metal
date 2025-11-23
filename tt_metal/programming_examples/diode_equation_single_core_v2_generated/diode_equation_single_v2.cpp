// HOST CODE
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

void golden_diode_equation(
    const std::vector<bfloat16>& voltage_vec, std::vector<bfloat16>& current_vec, float isat, float vj) {
    TT_FATAL(voltage_vec.size() == current_vec.size(), "Input and output vectors must be the same size");
    for (size_t i = 0; i < voltage_vec.size(); ++i) {
        float v = static_cast<float>(voltage_vec[i]);
        current_vec[i] = bfloat16(isat * (std::exp(v / vj) - 1.0f));
    }
}

inline float check_bfloat16_vector_pcc(const std::vector<bfloat16>& vec_a, const std::vector<bfloat16>& vec_b) {
    float x_mean = 0.0f;
    float y_mean = 0.0f;

    for (size_t i = 0; i < vec_a.size(); i++) {
        x_mean += static_cast<float>(vec_a[i]);
        y_mean += static_cast<float>(vec_b[i]);
    }

    x_mean /= vec_a.size();
    y_mean /= vec_a.size();

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
    y_stddev /= vec_a.size();

    return covariance / (std::sqrt(x_stddev) * std::sqrt(y_stddev));
}

int main() {
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    constexpr CoreCoord core = {0, 0};

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.f, 1.0f);

    std::vector<bfloat16> voltage_vec(constants::TILE_HW);
    for (bfloat16& v : voltage_vec) {
        v = bfloat16(dist(rng));
    }

    float isat = 1e-12f;
    float vj = 0.025f;

    std::vector<bfloat16> golden_current_vec(constants::TILE_HW, 0);
    golden_diode_equation(voltage_vec, golden_current_vec, isat, vj);

    voltage_vec = tilize_nfaces(voltage_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    constexpr uint32_t single_tile_size = sizeof(bfloat16) * constants::TILE_HEIGHT * constants::TILE_WIDTH;
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = sizeof(bfloat16) * voltage_vec.size()};
    std::shared_ptr<distributed::MeshBuffer> voltage_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> vj_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> current_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    distributed::EnqueueWriteMeshBuffer(cq, voltage_buffer, voltage_vec, false);

    constexpr uint32_t voltage_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_voltage_config =
        CircularBufferConfig(single_tile_size, {{voltage_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(voltage_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_voltage_config);

    constexpr uint32_t vj_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_vj_config =
        CircularBufferConfig(single_tile_size, {{vj_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(vj_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_vj_config);

    constexpr uint32_t current_cb_index = CBIndex::c_16;
    CircularBufferConfig cb_current_config =
        CircularBufferConfig(single_tile_size, {{current_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(current_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_current_config);

    std::vector<uint32_t> reader_compile_time_args = {voltage_cb_index, vj_cb_index};
    TensorAccessorArgs(*voltage_buffer).append_to(reader_compile_time_args);
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/diode_equation_single_core_v2_generated/kernels/dataflow/"
        "reader_binary_1_tile.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args = {current_cb_index};
    TensorAccessorArgs(*current_buffer).append_to(writer_compile_time_args);
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/diode_equation_single_core_v2_generated/kernels/dataflow/writer_1_tile.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});

    std::vector<uint32_t> compute_compile_time_args = {voltage_cb_index, vj_cb_index, current_cb_index};
    CreateKernel(
        program,
        "tt_metal/programming_examples/diode_equation_single_core_v2_generated/kernels/compute/diode_equation.cpp",
        core,
        tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});

    SetRuntimeArgs(program, reader_kernel_id, core, {voltage_buffer->address(), vj_buffer->address()});
    SetRuntimeArgs(program, writer_kernel_id, core, {current_buffer->address()});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<bfloat16> result_current_vec(constants::TILE_HW, 0);
    distributed::EnqueueReadMeshBuffer(cq, result_current_vec, current_buffer, true);

    result_current_vec = untilize_nfaces(result_current_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    const float pearson = check_bfloat16_vector_pcc(golden_current_vec, result_current_vec);
    fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
    TT_FATAL(pearson > 0.999, "PCC not high enough. Result PCC: {}, Expected PCC: 0.999", pearson);

    mesh_device->close();

    return 0;
}
