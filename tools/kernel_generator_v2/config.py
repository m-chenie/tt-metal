from pathlib import Path
import os

# Paths
TT_METAL_HOME = Path(os.environ.get("TT_METAL_HOME", "/home/m48chen/tt-metal"))
PROGRAMMING_EXAMPLES_DIR = TT_METAL_HOME / "tt_metal" / "programming_examples"
TOOLS_DIR = TT_METAL_HOME / "tools" / "kernel_generator_v2"
INDEX_PATH = TOOLS_DIR / "index.json"

# API headers to index for RAG
API_HEADER_DIRS = [
    TT_METAL_HOME / "tt_metal" / "include" / "compute_kernel_api",
    TT_METAL_HOME / "tt_metal" / "include" / "tt-metalium",
    TT_METAL_HOME / "tt_metal" / "hw" / "inc",
]

# SFPU-specific API headers (for focused retrieval)
# Includes: SFPU operations, circular buffers, registers, tile movement, dataflow
SFPU_HEADER_DIRS = [
    # SFPU compute operations
    TT_METAL_HOME / "tt_metal" / "include" / "compute_kernel_api" / "eltwise_binary_sfpu.h",
    TT_METAL_HOME / "tt_metal" / "include" / "compute_kernel_api" / "eltwise_unary",
    # Circular buffer and register management (compute kernel)
    TT_METAL_HOME / "tt_metal" / "include" / "compute_kernel_api" / "cb_api.h",
    TT_METAL_HOME / "tt_metal" / "include" / "compute_kernel_api" / "reg_api.h",
    TT_METAL_HOME / "tt_metal" / "include" / "compute_kernel_api" / "tile_move_copy.h",
    TT_METAL_HOME / "tt_metal" / "include" / "compute_kernel_api" / "pack.h",
    TT_METAL_HOME / "tt_metal" / "include" / "compute_kernel_api" / "common.h",
    # Dataflow APIs (reader/writer kernels)
    TT_METAL_HOME / "tt_metal" / "hw" / "inc" / "dataflow_api.h",
]

# Model / LLM
MODEL_DEFAULT = os.environ.get("TT_GEN_MODEL", "llama-3.3-70b-versatile")
TEMPERATURE = float(os.environ.get("TT_GEN_TEMP", 0.2))
MAX_TOKENS = int(os.environ.get("TT_GEN_MAX_TOKENS", 6000))

# Operations and core modes supported by prompts
OPERATIONS = {
    "add": {
        "description": "element-wise addition (C = A + B)",
        "compute_kernel": "add_tiles.cpp",
        "operation_type": "binary",
    },
    "subtract": {
        "description": "element-wise subtraction (C = A - B)",
        "compute_kernel": "subtract_tiles.cpp",
        "operation_type": "binary",
    },
    "multiply": {
        "description": "element-wise multiplication (C = A * B)",
        "compute_kernel": "multiply_tiles.cpp",
        "operation_type": "binary",
    },
    "matmul": {
        "description": "matrix multiplication (C = A @ B)",
        "compute_kernel": "matmul_tiles.cpp",
        "operation_type": "matmul",
    },
    "softplus": {
        "description": "softplus activation (C = log(1 + exp(A)))",
        "compute_kernel": "softplus_tiles.cpp",
        "operation_type": "sfpu_chain",
        "formula": "softplus(x) = log(1 + exp(x))",
        "mathematical_steps": "exponentiate input, add 1, take logarithm",
        "required_compute_functions": [
            "exp_tile",  # exponentiation
            "add_binary_tile",  # addition (exp(x) + 1)
            "log_tile",  # logarithm
        ],
    },
    "diode_equation": {
        "description": "diode current equation (I = isat × (exp(V/vj) - 1))",
        "compute_kernel": "diode_equation.cpp",
        "operation_type": "sfpu_chain",
        "inputs": ["V", "Vj", "Isat"],
        "variable_inputs": ["V"],  # Read from DRAM
        "constant_inputs": {  # Initialize in reader kernel using float_to_bfloat16
            "Vj": 1.0,
            "Isat": 0.026,
            "ones": 1.0,  # For subtraction (... - 1)
        },
        "formula": "I = isat × (exp(V/vj) - 1)",
        "mathematical_steps": "divide V by vj, exponentiate result, subtract 1, multiply by isat",
        "required_compute_functions": [
            "exp_tile",  # exponentiation
            "div_binary_tile",  # division (V/vj)
            "mul_binary_tile",  # multiplication (isat * ...)
            "sub_binary_tile",  # subtraction (... - 1)
        ],
    },
}

CORE_MODES = {
    "single": {"description": "single-core implementation", "suffix": "_single_core"},
    "multi": {"description": "multi-core implementation", "suffix": "_multi_core"},
}

# Indexing
SKIP_DIRS = {"build", "build_Release", "generated", "__pycache__", "__MACOSX"}
CODE_EXTS = {".cpp", ".h", ".hpp", ".cc"}
DOC_EXTS = {".md", ".rst", ".txt"}
MAX_FILE_SIZE = 15000  # Skip files larger than this (likely not example kernels)
TOP_K = 6  # retrieval fanout

# Canonical high-quality examples to prioritize
CANONICAL_EXAMPLES = {
    "sfpu_chain": {
        "compute": "tt_metal/programming_examples/sfpu_eltwise_chain/kernels/compute/compute.cpp",
        "reader": "tt_metal/programming_examples/sfpu_eltwise_chain/kernels/dataflow/reader.cpp",
        "writer": "tt_metal/programming_examples/sfpu_eltwise_chain/kernels/dataflow/writer.cpp",
        "host": "tt_metal/programming_examples/sfpu_eltwise_chain/sfpu_eltwise_chain.cpp",
    },
    "binary": {
        "compute": "tt_metal/programming_examples/eltwise_binary/kernels/compute/tiles_add.cpp",
        "reader": "tt_metal/programming_examples/eltwise_binary/kernels/dataflow/read_tiles.cpp",
        "writer": "tt_metal/programming_examples/eltwise_binary/kernels/dataflow/write_tile.cpp",
        "host": "tt_metal/programming_examples/eltwise_binary/eltwise_binary.cpp",
    },
    "matmul": {
        "compute": "tt_metal/programming_examples/matmul/matmul_single_core/kernels/compute/bmm.cpp",
        "reader": "tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/reader_bmm_tile_layout.cpp",
        "writer": "tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/writer_bmm_tile_layout.cpp",
        "host": "tt_metal/programming_examples/matmul/matmul_single_core/matmul_single_core.cpp",
    },
}


# Helper function to generate circular buffer layout dynamically
def generate_circular_buffer_layout(op_cfg: dict) -> dict:
    """
    Dynamically generate circular buffer layout for an operation.
    Creates one circular buffer per input (variable or constant) + one output buffer.

    Args:
        op_cfg: Operation configuration dict from OPERATIONS

    Returns:
        Dict mapping CB_N to buffer description with num_tiles
    """
    cb_layout = {}
    cb_index = 0

    # Add circular buffers for variable inputs (from DRAM)
    variable_inputs = op_cfg.get("variable_inputs", [])
    for var_input in variable_inputs:
        cb_id = f"CB_{cb_index}"
        cb_layout[cb_id] = {"description": f"{var_input} (variable input from DRAM)", "num_tiles": 1}
        cb_index += 1

    # Add circular buffers for constant inputs (one CB per constant)
    constant_inputs = op_cfg.get("constant_inputs", {})
    for const_name, const_val in constant_inputs.items():
        cb_id = f"CB_{cb_index}"
        cb_layout[cb_id] = {
            "description": f"{const_name} (constant = {const_val}, initialized in reader kernel)",
            "num_tiles": 1,
        }
        cb_index += 1

    # Add output circular buffer
    output_cb_id = f"CB_{cb_index}"
    cb_layout[output_cb_id] = {"description": "output result", "num_tiles": 1}

    return cb_layout


# Fallback CMakeLists.txt template if LLM generation fails
CMAKELISTS_TEMPLATE = """cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_{project_name})

add_executable(metal_example_{project_name})
target_sources(metal_example_{project_name} PRIVATE {source_file})

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_{project_name} PUBLIC TT::Metalium)
"""
