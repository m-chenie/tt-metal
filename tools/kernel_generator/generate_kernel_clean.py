#!/usr/bin/env python3
"""
TT-Metal Kernel Generator (Clean Version)
Based on the working legacy generator approach.

Workflow:
1. Generate kernels only (no host code, no CMakeLists.txt)
2. User manually creates host code and CMakeLists.txt
3. If compilation fails, use --refine flag with path to fix issues

RAG Examples:
- Single-core: add_2_integers_in_compute
- Multi-core: matmul_multi_core
"""

import os
import sys
import argparse
from pathlib import Path
from openai import OpenAI
from datetime import datetime
import re

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Supported operations
OPERATIONS = {
    "subtract": {
        "description": "subtraction (C = A - B)",
        "kernel_name": "subtract_2_tiles",
        "operation_symbol": "-",
    },
    "multiply": {
        "description": "multiplication (C = A * B)",
        "kernel_name": "multiply_2_tiles",
        "operation_symbol": "*",
    },
    "add": {
        "description": "addition (C = A + B)",
        "kernel_name": "add_2_tiles",
        "operation_symbol": "+",
    },
}

# Core modes
CORE_MODES = {
    "single": {
        "description": "single-core implementation",
        "suffix": "_single_core",
    },
    "multi": {
        "description": "multi-core implementation",
        "suffix": "_multi_core",
    },
}


def load_file(filepath):
    """Load a file and return its contents"""
    try:
        with open(filepath, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return ""


def build_rag_knowledge_base():
    """Build RAG knowledge base with add_2_integers_in_compute and matmul_multi_core"""
    tt_metal_home = Path("/home/m48chen/tt-metal")

    # Load single-core addition example (the working reference)
    add_compute = load_file(
        tt_metal_home / "tt_metal/programming_examples/add_2_integers_in_compute/kernels/compute/add_2_tiles.cpp"
    )
    add_reader = load_file(
        tt_metal_home
        / "tt_metal/programming_examples/add_2_integers_in_compute/kernels/dataflow/reader_binary_1_tile.cpp"
    )
    add_writer = load_file(
        tt_metal_home / "tt_metal/programming_examples/add_2_integers_in_compute/kernels/dataflow/writer_1_tile.cpp"
    )

    # Load multi-core matmul example (the working multi-core reference)
    matmul_host = load_file(
        tt_metal_home / "tt_metal/programming_examples/matmul/matmul_multi_core/matmul_multi_core.cpp"
    )
    matmul_compute = load_file(
        tt_metal_home / "tt_metal/programming_examples/matmul/matmul_multi_core/kernels/compute/mm.cpp"
    )
    matmul_reader = load_file(
        tt_metal_home
        / "tt_metal/programming_examples/matmul/matmul_multi_core/kernels/dataflow/reader_mm_output_tiles_partitioned.cpp"
    )
    matmul_writer = load_file(
        tt_metal_home
        / "tt_metal/programming_examples/matmul/matmul_multi_core/kernels/dataflow/writer_unary_interleaved_start_id.cpp"
    )

    # Load API headers
    eltwise_binary = load_file(tt_metal_home / "tt_metal/include/compute_kernel_api/eltwise_binary.h")
    cb_api = load_file(tt_metal_home / "tt_metal/include/compute_kernel_api/cb_api.h")

    return {
        "eltwise_binary_api": eltwise_binary,
        "cb_api": cb_api,
        # Single-core examples
        "add_compute": add_compute,
        "add_reader": add_reader,
        "add_writer": add_writer,
        # Multi-core examples
        "matmul_host": matmul_host,
        "matmul_compute": matmul_compute,
        "matmul_reader": matmul_reader,
        "matmul_writer": matmul_writer,
    }


def create_system_prompt_with_rag(knowledge_base, core_mode):
    """Create system prompt with RAG context"""

    api_docs = f"""## Available TT-Metal APIs

### Element-wise Binary Operations (from eltwise_binary.h):
- add_tiles_init(cb_in0, cb_in1) - Initialize addition
- add_tiles(cb_in0, cb_in1, itile0, itile1, idst) - Perform addition
- sub_tiles_init(cb_in0, cb_in1) - Initialize subtraction
- sub_tiles(cb_in0, cb_in1, itile0, itile1, idst) - Perform subtraction
- mul_tiles_init(cb_in0, cb_in1) - Initialize multiplication
- mul_tiles(cb_in0, cb_in1, itile0, itile1, idst) - Perform multiplication
- binary_op_init_common(cb_in0, cb_in1, cb_out) - Common initialization

### Circular Buffer Operations:
- cb_wait_front(cb_id, num_tiles) - Wait for input
- cb_reserve_back(cb_id, num_tiles) - Reserve output space
- cb_push_back(cb_id, num_tiles) - Push output
- cb_pop_front(cb_id, num_tiles) - Pop input

### Tile Register Management:
- tile_regs_acquire() - Acquire destination registers
- tile_regs_commit() - Commit to packer
- tile_regs_wait() - Wait for packer
- tile_regs_release() - Release registers
- pack_tile(dst_index, cb_out) - Pack tile to circular buffer"""

    if core_mode == "single":
        examples_section = f"""## Reference Implementation: Single-Core Addition Example

### Compute Kernel (add_2_tiles.cpp):
```cpp
{knowledge_base['add_compute']}
```

### Reader Kernel (reader_binary_1_tile.cpp):
```cpp
{knowledge_base['add_reader']}
```

### Writer Kernel (writer_1_tile.cpp):
```cpp
{knowledge_base['add_writer']}
```"""
    else:  # multi-core
        examples_section = f"""## Reference Implementation: Multi-Core Matmul Example

### Compute Kernel (mm.cpp):
```cpp
{knowledge_base['matmul_compute']}
```

### Reader Kernel (reader_mm_output_tiles_partitioned.cpp):
```cpp
{knowledge_base['matmul_reader']}
```

### Writer Kernel (writer_unary_interleaved_start_id.cpp):
```cpp
{knowledge_base['matmul_writer']}
```

## Single-Core Addition Example (for API reference):

### Compute Kernel (add_2_tiles.cpp):
```cpp
{knowledge_base['add_compute']}
```

### Reader Kernel (reader_binary_1_tile.cpp):
```cpp
{knowledge_base['add_reader']}
```

### Writer Kernel (writer_1_tile.cpp):
```cpp
{knowledge_base['add_writer']}
```"""

    return f"""You are an expert TT-Metal kernel developer for Tenstorrent's Wormhole architecture.

# KNOWLEDGE BASE (RAG Context)

{api_docs}

{examples_section}

# YOUR ROLE

You generate TT-Metal kernels by:
1. Understanding the pattern from the reference examples
2. Selecting the correct API functions for the requested operation
3. Creating your own implementation (don't copy verbatim)
4. Following TT-Metal best practices (proper headers, comments, error handling)
5. {"Using single-core patterns (simple loops, single tile processing)" if core_mode == "single" else "Using multi-core patterns (work distribution, core coordination, SPMD)"}

Generate clean, production-quality code."""


def generate_kernel(kernel_type, operation, operation_symbol, kernel_name, system_prompt, core_mode, model="gpt-4o"):
    """Generate a single kernel file"""

    if core_mode == "single":
        if kernel_type == "compute":
            user_prompt = f"""Generate a TT-Metal compute kernel for {operation} (C = A {operation_symbol} B).

Requirements:
- Find the {operation} API functions in the knowledge base
- Follow the single-core addition compute kernel pattern
- Use circular buffers: CB_0 (input A), CB_1 (input B), CB_16 (output)
- Single tile processing (1 tile at a time)
- Include comments explaining each step
- Filename: kernels/compute/{kernel_name}.cpp

Generate only the .cpp code."""

        elif kernel_type == "reader":
            user_prompt = f"""Generate a TT-Metal reader kernel for single-core {operation}.

Requirements:
- Follow the single-core addition reader kernel pattern
- Read two tiles from DRAM to CB_0 and CB_1
- Use NOC async reads with barriers
- Runtime args: arg 0 = src0 address, arg 1 = src1 address
- Filename: kernels/dataflow/reader_binary_1_tile.cpp

Generate only the .cpp code."""

        elif kernel_type == "writer":
            user_prompt = f"""Generate a TT-Metal writer kernel for single-core {operation}.

Requirements:
- Follow the single-core addition writer kernel pattern
- Write one tile from CB_16 to DRAM
- Use NOC async write with barrier
- Runtime arg: arg 0 = dst address
- Filename: kernels/dataflow/writer_1_tile.cpp

Generate only the .cpp code."""

    else:  # multi-core
        if kernel_type == "compute":
            user_prompt = f"""Generate a TT-Metal multi-core compute kernel for {operation} (C = A {operation_symbol} B).

Requirements:
- Find the {operation} API functions in the knowledge base
- Follow multi-core matmul compute kernel patterns for work distribution
- Use circular buffers: CB_0 (input A), CB_1 (input B), CB_16 (output)
- Handle multiple tiles per core (work distribution)
- Use runtime args to get work assignment
- Include proper headers and comments
- Filename: kernels/compute/{kernel_name}.cpp

Generate only the .cpp code."""

        elif kernel_type == "reader":
            user_prompt = f"""Generate a TT-Metal multi-core reader kernel for {operation}.

Requirements:
- Follow multi-core matmul reader kernel patterns
- Each core reads its assigned tiles from DRAM to CB_0 and CB_1
- Use proper address generation and NOC async reads
- Runtime args for work distribution
- For element-wise ops: read same tile_id from both sources
- Filename: kernels/dataflow/reader_binary_tiles_partitioned.cpp

Generate only the .cpp code."""

        elif kernel_type == "writer":
            user_prompt = f"""Generate a TT-Metal multi-core writer kernel for {operation}.

Requirements:
- Follow multi-core matmul writer kernel patterns
- Each core writes its assigned result tiles from CB_16 to DRAM
- Use proper address generation and NOC async writes
- Runtime args for work distribution
- Filename: kernels/dataflow/writer_tiles_partitioned.cpp

Generate only the .cpp code."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.2,
            max_tokens=3000,
        )

        generated_code = response.choices[0].message.content

        # Extract code from markdown if present
        if "```cpp" in generated_code:
            generated_code = generated_code.split("```cpp")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0].strip()

        return generated_code, {"user": user_prompt}

    except Exception as e:
        print(f"Error calling GPT-4 for {kernel_type}: {e}")
        return None, None


def refine_kernels_from_errors(example_path, model="gpt-4o"):
    """Refine kernels based on compilation errors"""
    example_dir = Path(example_path)

    if not example_dir.exists():
        print(f"Error: Example directory {example_path} does not exist")
        return False

    print(f"Refining kernels in {example_path}...")

    # TODO: Implement compilation error parsing and refinement
    # This would:
    # 1. Run cmake and make to get compilation errors
    # 2. Parse the errors
    # 3. Use GPT-4 to fix the kernels based on error messages
    # 4. Repeat until compilation succeeds or max iterations reached

    print("Refinement not yet implemented - manual fixes required for now")
    return False


def create_directory_structure(base_path):
    """Create kernel directory structure"""
    base = Path(base_path)
    base.mkdir(parents=True, exist_ok=True)
    (base / "kernels" / "compute").mkdir(parents=True, exist_ok=True)
    (base / "kernels" / "dataflow").mkdir(parents=True, exist_ok=True)
    return base


def save_generation_log(output_dir, operation, core_mode, model, all_prompts):
    """Save generation log"""
    log_file = output_dir / "GENERATION_LOG.md"

    content = f"""# TT-Metal Kernel Generation Log

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Operation:** {operation}
**Core Mode:** {core_mode}
**Model:** {model}

## Workflow
1. ✅ Generated kernels only (no host code, no CMakeLists.txt)
2. ⏳ Manually create host code and CMakeLists.txt
3. ⏳ If compilation fails, use --refine flag

## Generated Kernels
- compute kernel: kernels/compute/*.cpp
- reader kernel: kernels/dataflow/reader_*.cpp
- writer kernel: kernels/dataflow/writer_*.cpp

## Next Steps
1. Create CMakeLists.txt (copy from similar example)
2. Create host code (copy from similar example and adapt)
3. Build with cmake and make
4. If errors, run: ./generate_kernel.py --refine {output_dir}

---

"""

    for kernel_type, prompt_info in all_prompts.items():
        content += f"""## {kernel_type.upper()} Kernel Prompt

```
{prompt_info['user']}
```

---

"""

    log_file.write_text(content)
    return log_file


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="TT-Metal Kernel Generator (Clean Version)",
        epilog="Example: %(prog)s --operation add --core-mode single",
    )

    parser.add_argument("--operation", choices=list(OPERATIONS.keys()), help="Operation to generate kernels for")

    parser.add_argument("--core-mode", choices=list(CORE_MODES.keys()), help="Core mode: single or multi")

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: programming_examples/<op>_<mode>_generated)",
    )

    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model (default: gpt-4o)")

    parser.add_argument("--refine", type=str, help="Refine existing kernels at this path based on compilation errors")

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        return 1

    # Refinement mode
    if args.refine:
        success = refine_kernels_from_errors(args.refine, args.model)
        return 0 if success else 1

    # Generation mode - require operation and core_mode
    if not args.operation or not args.core_mode:
        print("ERROR: --operation and --core-mode required for generation")
        print("Use --refine <path> to refine existing kernels")
        return 1

    op_config = OPERATIONS[args.operation]
    core_config = CORE_MODES[args.core_mode]

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(
            f"/home/m48chen/tt-metal/tt_metal/programming_examples/{args.operation}{core_config['suffix']}_generated"
        )

    print("=" * 80)
    print("TT-Metal Kernel Generator (Clean Version)")
    print("=" * 80)
    print(f"Operation: {op_config['description']}")
    print(f"Core Mode: {core_config['description']}")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")

    print("\n[1/4] Building RAG knowledge base...")
    knowledge_base = build_rag_knowledge_base()
    print("   ✓ Loaded add_2_integers_in_compute example")
    print("   ✓ Loaded matmul_multi_core example")
    print("   ✓ Loaded API documentation")

    print(f"\n[2/4] Creating system prompt...")
    system_prompt = create_system_prompt_with_rag(knowledge_base, args.core_mode)
    print(f"   ✓ System prompt ready ({len(system_prompt)} chars)")

    print(f"\n[3/4] Creating directory structure...")
    base_dir = create_directory_structure(output_dir)
    print(f"   ✓ Created: {base_dir}")

    # Generate kernels
    if args.core_mode == "single":
        kernels_to_generate = [
            ("compute", f"kernels/compute/{op_config['kernel_name']}.cpp"),
            ("reader", "kernels/dataflow/reader_binary_1_tile.cpp"),
            ("writer", "kernels/dataflow/writer_1_tile.cpp"),
        ]
    else:  # multi-core
        kernels_to_generate = [
            ("compute", f"kernels/compute/{op_config['kernel_name']}.cpp"),
            ("reader", "kernels/dataflow/reader_binary_tiles_partitioned.cpp"),
            ("writer", "kernels/dataflow/writer_tiles_partitioned.cpp"),
        ]

    all_prompts = {}

    print(f"\n[4/4] Generating kernels...")
    for kernel_type, filepath in kernels_to_generate:
        print(f"   Generating {kernel_type}...")

        generated_code, prompt_info = generate_kernel(
            kernel_type,
            op_config["description"],
            op_config["operation_symbol"],
            op_config["kernel_name"],
            system_prompt,
            args.core_mode,
            model=args.model,
        )

        if not generated_code:
            print(f"   ✗ Failed to generate {kernel_type}")
            return 1

        output_path = base_dir / filepath
        output_path.write_text(generated_code)
        all_prompts[kernel_type] = prompt_info

        print(f"   ✓ {kernel_type}: {len(generated_code)} bytes -> {filepath}")

    # Save log
    log_file = save_generation_log(
        base_dir, op_config["description"], core_config["description"], args.model, all_prompts
    )
    print(f"   ✓ Log saved: {log_file}")

    print("\n" + "=" * 80)
    print("✅ KERNEL GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated {len(kernels_to_generate)} kernels at: {output_dir}")
    print("\nNext steps:")
    print("1. Create CMakeLists.txt (copy from similar example)")
    print("2. Create host code (copy from similar example and adapt)")
    print("3. Build with cmake and make")
    print("4. If compilation errors, run:")
    print(f"   ./generate_kernel_clean.py --refine {output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
