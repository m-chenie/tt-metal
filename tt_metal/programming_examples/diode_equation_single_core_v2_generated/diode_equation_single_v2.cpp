// HOST CODE: diode_equation_single_v2.cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>

#include <cmath>
#include <random>
#include <cstdint>
#include <vector>
#include <fmt/core.h>

using namespace tt;
using namespace tt::tt_metal;

// Diode equation: I = isat × (exp(V/vj) - 1)
void golden_diode_equation(const std::vector<float>& v, std::vector<float>& i, float isat, float vj) {
    for (size_t idx = 0; idx < v.size(); ++idx) {
        i[idx] = isat * (std::exp(v[idx] / vj) - 1.0f);
    }
}

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
    // Device setup
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);

    // Device command queue and program setup
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    // Core range setup
    constexpr CoreCoord core = {0, 0};

    // Input data preparation
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.f, 1.0f);

    // Fill the source vector with random values
    std::vector<bfloat16> src_vec(constants::TILE_HW);
    for (bfloat16& v : src_vec) {
        v = bfloat16(dist(rng));
    }

    // Calculate golden function results on CPU
    std::vector<float> golden_vec_float(src_vec.size());
    std::vector<float> src_vec_float(src_vec.size());
    for (size_t i = 0; i < src_vec.size(); ++i) {
        src_vec_float[i] = static_cast<float>(src_vec[i]);
    }
    golden_diode_equation(src_vec_float, golden_vec_float, 1.0f, 0.5f);

    // Tilize the input vectors to match the expected tiled layout for the device
    src_vec = tilize_nfaces(src_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // Dram buffer config
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * constants::TILE_HEIGHT * constants::TILE_WIDTH;
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = sizeof(bfloat16) * src_vec.size()};
    std::shared_ptr<distributed::MeshBuffer> src_dram_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());  // Input buffer
    std::shared_ptr<distributed::MeshBuffer> dst_dram_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());  // Output buffer

    // DRAM transfer
    distributed::EnqueueWriteMeshBuffer(cq, src_dram_buffer, src_vec, false);

    // L1 circular buffer setup
    constexpr uint32_t src_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src_config =
        CircularBufferConfig(single_tile_size, {{src_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src_config);

    constexpr uint32_t isat_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_isat_config =
        CircularBufferConfig(single_tile_size, {{isat_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(isat_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_isat_config);

    constexpr uint32_t result_cb_index = CBIndex::c_16;
    CircularBufferConfig cb_result_config =
        CircularBufferConfig(single_tile_size, {{result_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(result_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_result_config);

    // Kernels setup
    // Data movement kernels
    std::vector<uint32_t> reader_compile_time_args = {src_cb_index, isat_cb_index};
    TensorAccessorArgs(*src_dram_buffer).append_to(reader_compile_time_args);
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "diode_equation/kernels/dataflow/reader.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});
    std::vector<uint32_t> writer_compile_time_args = {result_cb_index};
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "diode_equation/kernels/dataflow/writer.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});

    // Compute kernel
    std::vector<uint32_t> compute_compile_time_args = {src_cb_index, isat_cb_index, result_cb_index};
    CreateKernel(
        program,
        "diode_equation/kernels/compute/compute.cpp",
        core,
        tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});

    // Runtime args setup
    SetRuntimeArgs(program, reader_kernel_id, core, {src_dram_buffer->address()});
    SetRuntimeArgs(program, writer_kernel_id, core, {dst_dram_buffer->address()});

    // Program enqueue
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Data transfer back to host machine
    std::vector<bfloat16> result_vec(src_vec.size(), 0);
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

    // Reverse the tilization to get the result in the row-major format that the CPU expects
    result_vec = untilize_nfaces(result_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // Calculate the Pearson correlation coefficient (PCC) between the golden vector and the result vector
    std::vector<bfloat16> golden_vec(result_vec.size());
    for (size_t i = 0; i < result_vec.size(); ++i) {
        golden_vec[i] = bfloat16(golden_vec_float[i]);
    }
    const float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
    fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
    TT_FATAL(pearson > 0.999, "PCC not high enough. Result PCC: {}, Expected PCC: 0.999", pearson);

    mesh_device->close();
}
