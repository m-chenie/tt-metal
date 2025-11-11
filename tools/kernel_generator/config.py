#!/usr/bin/env python3
"""
Configuration settings for TT-Metal Kernel Generator
"""

import os
from pathlib import Path

# Paths
TT_METAL_HOME = Path(os.environ.get("TT_METAL_HOME", "/home/m48chen/tt-metal"))
TOOLS_DIR = TT_METAL_HOME / "tools" / "kernel_generator"
EXAMPLES_DIR = TT_METAL_HOME / "tt_metal" / "programming_examples"
OUTPUT_DIR = TOOLS_DIR / "generated_examples"

# OpenAI / LLM Configuration
# Default model switched to working Groq model
OPENAI_MODEL_DEFAULT = "llama-3.3-70b-versatile"
OPENAI_TEMPERATURE = 0.2
OPENAI_MAX_TOKENS = 8000  # Increased to Llama 3.3's max output limit

# Supported Operations
OPERATIONS = {
    "add": {
        "description": "element-wise addition (C = A + B)",
        "kernel_name": "add_tiles",
        "operation_symbol": "+",
        "api_init": "add_tiles_init",
        "api_compute": "add_tiles",
    },
    "subtract": {
        "description": "element-wise subtraction (C = A - B)",
        "kernel_name": "subtract_tiles",
        "operation_symbol": "-",
        "api_init": "sub_tiles_init",
        "api_compute": "sub_tiles",
    },
    "multiply": {
        "description": "element-wise multiplication (C = A * B)",
        "kernel_name": "multiply_tiles",
        "operation_symbol": "*",
        "api_init": "mul_tiles_init",
        "api_compute": "mul_tiles",
    },
    "matmul": {
        "description": "matrix multiplication (C = A @ B)",
        "kernel_name": "matmul_tiles",
        "operation_symbol": "@",
        "api_init": "mm_init",
        "api_compute": "matmul_tiles",
    },
    # SFPU Eltwise Chain Operations
    "softplus": {
        "description": "softplus activation (C = log(1 + exp(A)))",
        "kernel_name": "softplus_tiles",
        "operation_symbol": "softplus",
        "api_init": ["exp_tile_init", "add_binary_tile_init", "log_tile_init"],
        "api_compute": ["exp_tile", "add_binary_tile", "log_tile"],
        "operation_type": "sfpu_chain",
        "requires_constants": True,
        "constants": {"ones": 1.0},
    },
    "diode_equation": {
        "description": "diode current equation (I = isat × (exp(V/vj) - 1))",
        "kernel_name": "diode_equation",
        "operation_symbol": "diode",
        "api_init": ["div_binary_tile_init", "exp_tile_init", "sub_binary_tile_init", "mul_binary_tile_init"],
        "api_compute": ["div_binary_tile", "exp_tile", "sub_binary_tile", "mul_binary_tile"],
        "operation_type": "sfpu_chain",
        "requires_constants": True,
        "constants": {"ones": 1.0},
        "inputs": ["V", "vj", "isat"],
        "formula": "isat × (exp(V/vj) - 1)",
    },
}

# Core Modes
CORE_MODES = {
    "single": {
        "description": "single-core implementation",
        "suffix": "_single_core",
        "max_cores": 1,
    },
    "multi": {
        "description": "multi-core implementation",
        "suffix": "_multi_core",
        "max_cores": None,  # Use all available cores
    },
}

# RAG Knowledge Base Sources
RAG_SOURCES = {
    "single_core_examples": [
        "add_2_integers_in_compute",
        "eltwise_binary",  # Binary FPU operations
        "eltwise_sfpu",  # Single operation SFPU
    ],
    "sfpu_chain_examples": [
        "sfpu_eltwise_chain",
    ],
    "multi_core_examples": [
        "matmul/matmul_multi_core",
        "vecadd_multi_core",
    ],
    "api_headers": [
        "tt_metal/include/compute_kernel_api/eltwise_binary.h",
        "tt_metal/include/compute_kernel_api/eltwise_binary_sfpu.h",
        "tt_metal/include/compute_kernel_api/eltwise_unary/eltwise_unary.h",
        "tt_metal/include/compute_kernel_api/eltwise_unary/exp.h",
        "tt_metal/include/compute_kernel_api/cb_api.h",
        "tt_metal/include/compute_kernel_api/matmul.h",
        "tt_metal/include/compute_kernel_api/tile_move_copy.h",
        "tt_metal/include/compute_kernel_api/common.h",
    ],
    "host_api_headers": [
        "tt_metal/include/tt-metalium/host_api.hpp",
        "tt_metal/include/tt-metalium/work_split.hpp",
        "tt_metal/include/tt-metalium/distributed.hpp",
    ],
}

# Compilation Settings
COMPILE_TIMEOUT = 120  # seconds
MAX_REFINEMENT_ITERATIONS = 2
BUILD_TYPE = "Release"

# Validation Settings
VALIDATION_TIMEOUT = 300  # seconds
TEST_MATRIX_SIZES = [(64, 64, 64), (128, 128, 128)]  # M, N, K for matmul
TEST_TOLERANCE = 1e-2

# Logging
LOG_LEVEL = "INFO"
SAVE_PROMPTS = True
SAVE_COMPILATION_LOGS = True
