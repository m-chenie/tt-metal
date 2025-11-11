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
 * @brief Computes the diode equation: isat * (exp(V/vj) - 1)
 */
void golden_diode_equation(
    const std::vector<bfloat16>& V_vec,
    const std::vector<bfloat16>& vj_vec,
    const std::vector<bfloat16>& isat_vec,
    std::vector<bfloat16>& result_vec) {
    TT_FATAL(
        V_vec.size() == vj_vec.size() && V_vec.size() == isat_vec.size() && V_vec.size() == result_vec.size(),
        "All vectors must be the same size");

    for (size_t i = 0; i < V_vec.size(); ++i) {
        float V = static_cast<float>(V_vec[i]);
        float vj = static_cast<float>(vj_vec[i]);
        float isat = static_cast<float>(isat_vec[i]);

        // Diode equation: isat * (exp(V/vj) - 1)
        float result = isat * (std::exp(V / vj) - 1.0f);
        result_vec[i] = bfloat16(result);
    }
}

/**
 * @brief Calculates the Pearson Correlation Coefficient between two vectors
 */
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

    float correlation_coefficient = covariance / (std::sqrt(x_stddev) * std::sqrt(y_stddev));
    return correlation_coefficient;
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

    // Input data preparation - realistic diode parameters
    std::mt19937 rng(std::random_device{}());

    // Voltage: 0.1V to 1.5V range
    std::uniform_real_distribution<float> V_dist(0.1f, 1.5f);
    std::vector<bfloat16> V_vec(constants::TILE_HW);
    for (bfloat16& v : V_vec) {
        v = bfloat16(V_dist(rng));
    }

    // Junction voltage: typically around 0.7V for silicon diodes
    std::uniform_real_distribution<float> vj_dist(0.6f, 0.8f);
    std::vector<bfloat16> vj_vec(constants::TILE_HW);
    for (bfloat16& v : vj_vec) {
        v = bfloat16(vj_dist(rng));
    }

    // Saturation current: very small values
    std::uniform_real_distribution<float> isat_dist(1e-6f, 1e-4f);  // Adjusted for bfloat16 range
    std::vector<bfloat16> isat_vec(constants::TILE_HW);
    for (bfloat16& v : isat_vec) {
        v = bfloat16(isat_dist(rng));
    }

    // Calculate golden results on CPU
    std::vector<bfloat16> golden_vec(constants::TILE_HW, 0);
    golden_diode_equation(V_vec, vj_vec, isat_vec, golden_vec);

    // Tilize input vectors
    V_vec = tilize_nfaces(V_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);
    vj_vec = tilize_nfaces(vj_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);
    isat_vec = tilize_nfaces(isat_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // DRAM buffer config
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * constants::TILE_HEIGHT * constants::TILE_WIDTH;
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = single_tile_size};

    std::shared_ptr<distributed::MeshBuffer> V_dram_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> vj_dram_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> isat_dram_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> dst_dram_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    // DRAM transfers
    distributed::EnqueueWriteMeshBuffer(cq, V_dram_buffer, V_vec, false);
    distributed::EnqueueWriteMeshBuffer(cq, vj_dram_buffer, vj_vec, false);
    distributed::EnqueueWriteMeshBuffer(cq, isat_dram_buffer, isat_vec, false);

    // L1 circular buffer setup
    constexpr uint32_t V_cb_index = CBIndex::c_0;
    constexpr uint32_t vj_cb_index = CBIndex::c_1;
    constexpr uint32_t isat_cb_index = CBIndex::c_2;
    constexpr uint32_t ones_cb_index = CBIndex::c_3;
    constexpr uint32_t result_cb_index = CBIndex::c_16;

    CircularBufferConfig cb_V_config = CircularBufferConfig(single_tile_size, {{V_cb_index, tt::DataFormat::Float16_b}})
                                           .set_page_size(V_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_V_config);

    CircularBufferConfig cb_vj_config =
        CircularBufferConfig(single_tile_size, {{vj_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(vj_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_vj_config);

    CircularBufferConfig cb_isat_config =
        CircularBufferConfig(single_tile_size, {{isat_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(isat_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_isat_config);

    CircularBufferConfig cb_ones_config =
        CircularBufferConfig(single_tile_size, {{ones_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ones_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_ones_config);

    CircularBufferConfig cb_result_config =
        CircularBufferConfig(single_tile_size, {{result_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(result_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_result_config);

    // Create kernels
    std::vector<uint32_t> reader_compile_time_args = {V_cb_index, vj_cb_index, isat_cb_index, ones_cb_index};
    TensorAccessorArgs(*V_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*vj_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*isat_dram_buffer).append_to(reader_compile_time_args);

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "diode_equation_single_core_generated/kernels/dataflow/reader_binary_1_tile.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args = {result_cb_index};
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "diode_equation_single_core_generated/kernels/dataflow/writer_1_tile.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    std::vector<uint32_t> compute_compile_time_args = {
        V_cb_index, vj_cb_index, isat_cb_index, ones_cb_index, result_cb_index};

    [[maybe_unused]] KernelHandle compute_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "diode_equation_single_core_generated/kernels/compute/diode_equation.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    // Set runtime arguments
    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        {V_dram_buffer->address(), vj_dram_buffer->address(), isat_dram_buffer->address()});

    SetRuntimeArgs(program, writer_kernel_id, core, {dst_dram_buffer->address()});

    // Execute program
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Read results
    std::vector<bfloat16> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

    // Untilize results
    result_vec = untilize_nfaces(result_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // Calculate PCC
    float pcc = check_bfloat16_vector_pcc(golden_vec, result_vec);
    log_info(LogTest, "PCC: {}", pcc);

    // Check if test passed (PCC > 0.99 indicates high correlation)
    bool pass = pcc > 0.99f;

    // Close device
    pass &= mesh_device->close();

    if (pass) {
        log_info(LogTest, "Test Passed");
        return 0;
    } else {
        log_fatal(LogTest, "Test Failed: PCC = {}", pcc);
        return 1;
    }
}
