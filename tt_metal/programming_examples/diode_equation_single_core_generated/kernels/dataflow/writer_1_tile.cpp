#include <cstdint>
void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    cb_wait_front(cb_out0, 1);
    uint32_t cb_out0_addr = get_read_ptr(cb_out0);
    noc_async_write_tile(0, dst_addr, cb_out0_addr);
    noc_async_write_barrier();
    cb_pop_front(cb_out0, 1);
}
