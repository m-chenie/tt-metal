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
#include <iostream>
#include <chrono>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

/**
 * @brief Computes the diode current equation element-wise on input vectors
 *
 * The diode current equation is defined as: I = isat × (exp(V/vj) - 1)
 *
 * @param v_vec Input vector containing bfloat16 voltage values
 * @param isat_vec Input vector containing bfloat16 saturation current values
 * @param vj_vec Input vector containing bfloat16 thermal voltage values
 * @param result_vec Output vector where computed current values will be stored
 *                   Must be the same size as input vectors
 *
 * @throws TT_FATAL if input and output vectors have different sizes
 */
void golden_diode_current(
    const std::vector<bfloat16>& v_vec,
    const std::vector<bfloat16>& isat_vec,
    const std::vector<bfloat16>& vj_vec,
    std::vector<bfloat16>& result_vec) {
    TT_FATAL(v_vec.size() == result_vec.size(), "Input and output vectors must be the same size");
    for (size_t i = 0; i < v_vec.size(); ++i) {
        float v = static_cast<float>(v_vec[i]);
        float isat = static_cast<float>(isat_vec[i]);
        float vj = static_cast<float>(vj_vec[i]);
        result_vec[i] = bfloat16(isat * (std::exp(v / vj) - 1.0f));
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
float check_bfloat16_vector_pcc(const std::vector<bfloat16>& vec_a, const std::vector<bfloat16>& vec_b) {
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
    try {
        // Device setup
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram();

        // Core range setup
        constexpr CoreCoord core = {0, 0};

        // Input data preparation
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.1f, 1.0f);

        // Fill the source vectors with random values
        std::vector<bfloat16> v_vec(constants::TILE_HW);
        std::vector<bfloat16> isat_vec(constants::TILE_HW);
        std::vector<bfloat16> vj_vec(constants::TILE_HW);
        for (bfloat16& v : v_vec) {
            v = bfloat16(dist(rng));
        }
        for (bfloat16& isat : isat_vec) {
            isat = bfloat16(dist(rng));
        }
        for (bfloat16& vj : vj_vec) {
            vj = bfloat16(dist(rng));
        }

        // Calculate golden function results on CPU
        std::vector<bfloat16> golden_vec(constants::TILE_HW, 0);
        golden_diode_current(v_vec, isat_vec, vj_vec, golden_vec);

        // Tilize the input vectors to match the expected tiled layout for the device
        v_vec = tilize_nfaces(v_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);
        isat_vec = tilize_nfaces(isat_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);
        vj_vec = tilize_nfaces(vj_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

        // DRAM buffer config
        constexpr uint32_t single_tile_size = sizeof(bfloat16) * constants::TILE_HEIGHT * constants::TILE_WIDTH;
        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};
        distributed::ReplicatedBufferConfig buffer_config{.size = sizeof(bfloat16) * v_vec.size()};
        std::shared_ptr<distributed::MeshBuffer> v_dram_buffer =
            distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        std::shared_ptr<distributed::MeshBuffer> isat_dram_buffer =
            distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        std::shared_ptr<distributed::MeshBuffer> vj_dram_buffer =
            distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        std::shared_ptr<distributed::MeshBuffer> dst_dram_buffer =
            distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

        // DRAM transfer
        distributed::EnqueueWriteMeshBuffer(cq, v_dram_buffer, v_vec, false);
        distributed::EnqueueWriteMeshBuffer(cq, isat_dram_buffer, isat_vec, false);
        distributed::EnqueueWriteMeshBuffer(cq, vj_dram_buffer, vj_vec, false);

        // L1 circular buffer setup
        constexpr uint32_t v_cb_index = CBIndex::c_0;
        CircularBufferConfig cb_v_config =
            CircularBufferConfig(single_tile_size, {{v_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(v_cb_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_v_config);

        constexpr uint32_t isat_cb_index = CBIndex::c_1;
        CircularBufferConfig cb_isat_config =
            CircularBufferConfig(single_tile_size, {{isat_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(isat_cb_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_isat_config);

        constexpr uint32_t vj_cb_index = CBIndex::c_2;
        CircularBufferConfig cb_vj_config =
            CircularBufferConfig(single_tile_size, {{vj_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(vj_cb_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_vj_config);

        constexpr uint32_t result_cb_index = CBIndex::c_3;
        CircularBufferConfig cb_result_config =
            CircularBufferConfig(single_tile_size, {{result_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(result_cb_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_result_config);

        // Kernels setup
        std::vector<uint32_t> reader_compile_time_args = {v_cb_index, isat_cb_index, vj_cb_index};
        TensorAccessorArgs(*v_dram_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*isat_dram_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*vj_dram_buffer).append_to(reader_compile_time_args);
        KernelHandle reader_kernel_id = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "diode_equation/kernels/dataflow/reader_binary_1_tile.cpp",
            core,
            tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});

        std::vector<uint32_t> writer_compile_time_args = {result_cb_index};
        TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
        KernelHandle writer_kernel_id = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "diode_equation/kernels/dataflow/writer_1_tile.cpp",
            core,
            tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});

        std::vector<uint32_t> compute_compile_time_args = {v_cb_index, isat_cb_index, vj_cb_index, result_cb_index};
        CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "diode_equation/kernels/compute/diode_equation.cpp",
            core,
            tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});

        // Runtime args setup
        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {v_dram_buffer->address(), isat_dram_buffer->address(), vj_dram_buffer->address()});
        SetRuntimeArgs(program, writer_kernel_id, core, {dst_dram_buffer->address()});

        // Program enqueue
        auto start = std::chrono::high_resolution_clock::now();
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

        // Data transfer back to host machine
        std::vector<bfloat16> result_vec(constants::TILE_HW, 0);
        distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

        // Reverse the tilization to get the result in the row-major format that the CPU expects
        result_vec = untilize_nfaces(result_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

        // Validate results
        const float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
        std::cout << "Metalium vs Golden -- PCC = " << pearson << std::endl;
        TT_FATAL(
            pearson > 0.999, "PCC not high enough. Result PCC: " + std::to_string(pearson) + ", Expected PCC: 0.999");

        mesh_device->close();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
