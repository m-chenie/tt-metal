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

using namespace tt;
using namespace tt::tt_metal;

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
    std::vector<bfloat16> V_vec(constants::TILE_HW);
    for (bfloat16& v : V_vec) {
        v = bfloat16(dist(rng));
    }

    std::vector<bfloat16> Vj_vec(constants::TILE_HW, bfloat16(1.0f));      // Vj is a constant
    std::vector<bfloat16> Isat_vec(constants::TILE_HW, bfloat16(0.026f));  // Isat is a constant

    // Save untilized data for golden calculation
    std::vector<bfloat16> V_untilized = V_vec;
    std::vector<bfloat16> Vj_untilized = Vj_vec;
    std::vector<bfloat16> Isat_untilized = Isat_vec;

    // Tilize the input vectors to match the expected tiled layout for the device
    V_vec = tilize_nfaces(V_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);
    Vj_vec = tilize_nfaces(Vj_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);
    Isat_vec = tilize_nfaces(Isat_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // Dram buffer config
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * constants::TILE_HEIGHT * constants::TILE_WIDTH;
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = sizeof(bfloat16) * V_vec.size()};
    std::shared_ptr<distributed::MeshBuffer> V_dram_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());  // Input buffer
    std::shared_ptr<distributed::MeshBuffer> Vj_dram_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());  // Input buffer
    std::shared_ptr<distributed::MeshBuffer> Isat_dram_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());  // Input buffer
    std::shared_ptr<distributed::MeshBuffer> I_dram_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());  // Output buffer

    // DRAM transfer
    distributed::EnqueueWriteMeshBuffer(cq, V_dram_buffer, V_vec, false);
    distributed::EnqueueWriteMeshBuffer(cq, Vj_dram_buffer, Vj_vec, false);
    distributed::EnqueueWriteMeshBuffer(cq, Isat_dram_buffer, Isat_vec, false);

    // L1 circular buffer setup
    constexpr uint32_t V_cb_index = CBIndex::c_0;
    CircularBufferConfig V_cb_config = CircularBufferConfig(single_tile_size, {{V_cb_index, tt::DataFormat::Float16_b}})
                                           .set_page_size(V_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, V_cb_config);

    constexpr uint32_t Vj_cb_index = CBIndex::c_1;
    CircularBufferConfig Vj_cb_config =
        CircularBufferConfig(single_tile_size, {{Vj_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(Vj_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, Vj_cb_config);

    constexpr uint32_t Isat_cb_index = CBIndex::c_2;
    CircularBufferConfig Isat_cb_config =
        CircularBufferConfig(single_tile_size, {{Isat_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(Isat_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, Isat_cb_config);

    constexpr uint32_t ones_cb_index = CBIndex::c_3;
    CircularBufferConfig ones_cb_config =
        CircularBufferConfig(single_tile_size, {{ones_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ones_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, ones_cb_config);

    constexpr uint32_t result_cb_index = CBIndex::c_4;
    CircularBufferConfig result_cb_config =
        CircularBufferConfig(single_tile_size, {{result_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(result_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, result_cb_config);

    // Kernels setup
    std::vector<uint32_t> reader_compile_time_args = {V_cb_index, Vj_cb_index, Isat_cb_index, ones_cb_index};
    TensorAccessorArgs(*V_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*Vj_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*Isat_dram_buffer).append_to(reader_compile_time_args);
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/diode_equation_single_core_v2_generated/kernels/dataflow/"
        "reader_binary_1_tile.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args = {result_cb_index};
    TensorAccessorArgs(*I_dram_buffer).append_to(writer_compile_time_args);
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/diode_equation_single_core_v2_generated/kernels/dataflow/writer_1_tile.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});

    std::vector<uint32_t> compute_compile_time_args = {
        V_cb_index, Vj_cb_index, Isat_cb_index, ones_cb_index, result_cb_index};
    CreateKernel(
        program,
        "tt_metal/programming_examples/diode_equation_single_core_v2_generated/kernels/compute/diode_equation.cpp",
        core,
        tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});

    // Runtime args setup
    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        {V_dram_buffer->address(), Vj_dram_buffer->address(), Isat_dram_buffer->address()});
    SetRuntimeArgs(program, writer_kernel_id, core, {I_dram_buffer->address()});

    // Program enqueue
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Data transfer back to host machine
    std::vector<bfloat16> I_vec(constants::TILE_HW, 0);
    distributed::EnqueueReadMeshBuffer(cq, I_vec, I_dram_buffer, true);

    // Reverse the tilization to get the result in the row-major format that the CPU expects
    I_vec = untilize_nfaces(I_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // Calculate the golden function results on CPU using untilized data
    std::vector<bfloat16> golden_vec(constants::TILE_HW, 0);
    for (size_t i = 0; i < constants::TILE_HW; ++i) {
        float V = static_cast<float>(V_untilized[i]);
        float Vj = static_cast<float>(Vj_untilized[i]);
        float Isat = static_cast<float>(Isat_untilized[i]);
        golden_vec[i] = bfloat16(Isat * (std::exp(V / Vj) - 1.0f));
    }

    // Calculate the Pearson correlation coefficient (PCC) between the golden vector and the result vector
    float pearson = 0.0f;
    float x_mean = 0.0f;
    float y_mean = 0.0f;
    float covariance = 0.0f;
    float x_stddev = 0.0f;
    float y_stddev = 0.0f;

    for (size_t i = 0; i < constants::TILE_HW; i++) {
        float x_diff = static_cast<float>(I_vec[i]) - x_mean;
        float y_diff = static_cast<float>(golden_vec[i]) - y_mean;

        covariance += x_diff * y_diff;
        x_stddev += x_diff * x_diff;
        y_stddev += y_diff * y_diff;
    }

    covariance /= constants::TILE_HW;
    x_stddev /= constants::TILE_HW;
    y_stddev /= constants::TILE_HW;

    pearson = covariance / (std::sqrt(x_stddev) * std::sqrt(y_stddev));

    fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
    TT_FATAL(pearson > 0.999, "PCC not high enough. Result PCC: {}, Expected PCC: 0.999", pearson);

    mesh_device->close();
}
