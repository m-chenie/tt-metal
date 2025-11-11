#include <cstdint>
void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    cb_reserve_back(cb_in0, 1);
    cb_reserve_back(cb_in1, 1);
    uint32_t cb_in0_addr = get_write_ptr(cb_in0);
    uint32_t cb_in1_addr = get_write_ptr(cb_in1);
    noc_async_read_tile(0, src0_addr, cb_in0_addr);
    noc_async_read_tile(0, src1_addr, cb_in1_addr);
    noc_async_read_barrier();
    cb_push_back(cb_in0, 1);
    cb_push_back(cb_in1, 1);
}
