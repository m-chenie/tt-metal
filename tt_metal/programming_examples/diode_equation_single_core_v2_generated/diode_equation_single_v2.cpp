// HOST CODE
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

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

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
    std::vector<bfloat16> golden_vec(constants::TILE_HW, 0);
    for (size_t i = 0; i < src_vec.size(); ++i) {
        float isat = 1.0f;  // saturation current
        float vj = 1.0f;    // thermal voltage
        float v = static_cast<float>(src_vec[i]);
        golden_vec[i] = bfloat16(isat * (std::exp(v / vj) - 1));
    }

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

    constexpr uint32_t result_cb_index = CBIndex::c_16;
    CircularBufferConfig cb_result_config =
        CircularBufferConfig(single_tile_size, {{result_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(result_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_result_config);

    // Kernels setup
    // Data movement kernels
    std::vector<uint32_t> reader_compile_time_args = {src_cb_index};
    TensorAccessorArgs(*src_dram_buffer).append_to(reader_compile_time_args);
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "diode_equation_single_core_v2_generated/kernels/dataflow/reader_binary_1_tile.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});
    std::vector<uint32_t> writer_compile_time_args = {result_cb_index};
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "diode_equation_single_core_v2_generated/kernels/dataflow/writer_1_tile.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});

    // Compute kernel
    std::vector<uint32_t> compute_compile_time_args = {src_cb_index, result_cb_index};
    CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "diode_equation_single_core_v2_generated/kernels/compute/diode_equation.cpp",
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
    std::vector<bfloat16> result_vec(constants::TILE_HW, 0);
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

    // Reverse the tilization to get the result in the row-major format that the CPU expects
    result_vec = untilize_nfaces(result_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // Calculate the Pearson correlation coefficient (PCC) between the golden vector and the result vector
    // This is a measure of how similar the two vectors are.
    // A PCC close to 1 indicates that the two vectors are very similar.
    float pearson = 0.0f;
    float x_mean = 0.0f;
    float y_mean = 0.0f;
    float covariance = 0.0f;
    float x_stddev = 0.0f;
    float y_stddev = 0.0f;
    for (size_t i = 0; i < golden_vec.size(); i++) {
        x_mean += static_cast<float>(golden_vec[i]);
        y_mean += static_cast<float>(result_vec[i]);
    }
    x_mean /= golden_vec.size();
    y_mean /= result_vec.size();

    for (size_t i = 0; i < golden_vec.size(); i++) {
        float x_diff = static_cast<float>(golden_vec[i]) - x_mean;
        float y_diff = static_cast<float>(result_vec[i]) - y_mean;

        covariance += x_diff * y_diff;
        x_stddev += x_diff * x_diff;
        y_stddev += y_diff * y_diff;
    }

    covariance /= golden_vec.size();
    x_stddev /= golden_vec.size();
    y_stddev /= result_vec.size();

    pearson = covariance / (std::sqrt(x_stddev) * std::sqrt(y_stddev));
    fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
    TT_FATAL(pearson > 0.999, "PCC not high enough. Result PCC: {}, Expected PCC: 0.999", pearson);

    mesh_device->close();
}
