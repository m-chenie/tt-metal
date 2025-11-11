#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/exp.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    // Initialize SFPU operations
    exp_tile_init();
    add_binary_tile_init();
    sub_binary_tile_init();
    mul_binary_tile_init();
    div_binary_tile_init();

    // Wait for input tiles
    cb_wait_front(cb_vj, 1);
    cb_wait_front(cb_isat, 1);

    // Copy input tiles to registers
    tile_regs_acquire();
    copy_tile(cb_vj, /*offset*/ 0, /*register_offset*/ 0);
    copy_tile(cb_isat, /*offset*/ 0, /*register_offset*/ 1);

    // Calculate exp(V/vj)
    div_binary_tile(0, 1, 2);  // V/vj
    exp_tile(2);               // exp(V/vj)

    // Calculate exp(V/vj) - 1
    bfloat16 one = bfloat16(1.0f);
    sub_binary_tile(2, one, 4);  // exp(V/vj) - 1

    // Calculate isat * (exp(V/vj) - 1)
    mul_binary_tile(1, 4, 0);  // isat * (exp(V/vj) - 1)

    // Copy result to output tile
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);

    // Release registers and push output tile
    tile_regs_release();
    cb_push_back(cb_out, 1);

    // Pop input tiles
    cb_pop_front(cb_vj, 1);
    cb_pop_front(cb_isat, 1);
}  // namespace NAMESPACE
