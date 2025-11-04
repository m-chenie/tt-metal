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

# OpenAI Configuration
OPENAI_MODEL_DEFAULT = "gpt-4o"
OPENAI_TEMPERATURE = 0.2
OPENAI_MAX_TOKENS = 4000

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
        "subtract_2_integers_in_compute_llm",
    ],
    "multi_core_examples": [
        "matmul/matmul_multi_core",
        "subtract_multi_core_kernels_llm",
        "vecadd_multi_core",
    ],
    "api_headers": [
        "tt_metal/include/compute_kernel_api/eltwise_binary.h",
        "tt_metal/include/compute_kernel_api/cb_api.h",
        "tt_metal/include/compute_kernel_api/matmul.h",
        "tt_metal/include/compute_kernel_api/tile_move_copy.h",
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
