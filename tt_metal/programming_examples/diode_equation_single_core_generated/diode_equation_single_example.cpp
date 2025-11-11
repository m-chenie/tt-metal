#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>
using namespace tt::tt_metal;
int main(int /*argc*/, char** /*argv*/) {
    // Create a device object
    Device device = CreateDevice();

    // Create a program object
    Program program = CreateProgram();

    // Create kernel handles
    KernelHandle reader_kernel_id = CreateKernel(tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});
    KernelHandle writer_kernel_id = CreateKernel(tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});
    KernelHandle compute_kernel_id =
        CreateKernel(tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});

    // Set runtime arguments
    SetRuntimeArgs(program, reader_kernel_id, device, {src0_dram_buffer->address(), src1_dram_buffer->address()});
    SetRuntimeArgs(program, writer_kernel_id, device, {dst_dram_buffer->address()});
    SetRuntimeArgs(program, compute_kernel_id, device, {});

    return 0;
}
