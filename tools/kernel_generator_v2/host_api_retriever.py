"""
Host API function signature retrieval.

Extracts host API functions from host code examples and retrieves their signatures
from TT-Metal host headers (tt-metalium/*.hpp).
"""

import re
from pathlib import Path
from typing import List, Dict, Set, Any, Optional
from config import TT_METAL_HOME


# Common host API patterns specific to TT-Metal host code
HOST_API_PATTERNS = {
    # MeshDevice and distributed APIs
    "distributed": [
        "distributed::MeshDevice::create_unit_mesh",
        "distributed::MeshDevice::create",
        "distributed::MeshBuffer::create",
        "distributed::EnqueueWriteMeshBuffer",
        "distributed::EnqueueReadMeshBuffer",
        "distributed::EnqueueMeshWorkload",
        "distributed::Finish",
    ],
    # Program and kernel creation
    "program": [
        "CreateProgram",
        "CreateKernel",
        "SetRuntimeArgs",
    ],
    # Circular buffer APIs
    "circular_buffer": [
        "CreateCircularBuffer",
        "CircularBufferConfig",
    ],
    # Data movement
    "data": [
        "TensorAccessorArgs",
        "tilize_nfaces",
        "untilize_nfaces",
    ],
    # Device and core
    "device": [
        "Device::create",
        "Device::close",
    ],
}


def extract_host_api_calls_from_code(code: str) -> Set[str]:
    """
    Parse host C++ code and extract TT-Metal host API function calls.

    Looks for patterns like:
    - distributed::MeshDevice::create_unit_mesh(...)
    - CreateProgram()
    - CreateKernel(...)
    - SetRuntimeArgs(...)

    Returns set of function names (including namespace/class qualifiers).
    """
    api_calls = set()

    # Pattern 1: Namespace/class qualified calls (distributed::MeshDevice::create_unit_mesh)
    qualified_pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*(?:::[a-zA-Z_][a-zA-Z0-9_]*)+)\s*\("
    qualified_matches = re.findall(qualified_pattern, code)
    api_calls.update(qualified_matches)

    # Pattern 2: Simple function calls (CreateProgram, CreateKernel, etc.)
    # Focus on Create*, Set*, Enqueue* patterns common in TT-Metal
    simple_pattern = r"\b((?:Create|Set|Enqueue|tilize|untilize)[A-Z][a-zA-Z0-9_]*)\s*\("
    simple_matches = re.findall(simple_pattern, code)
    api_calls.update(simple_matches)

    # Pattern 3: Class constructors (CircularBufferConfig)
    constructor_pattern = r"\b([A-Z][a-zA-Z0-9_]*(?:Config|Args|Range))\s*\("
    constructor_matches = re.findall(constructor_pattern, code)
    api_calls.update(constructor_matches)

    return api_calls


def find_host_function_signature(func_name: str, header_dirs: List[Path]) -> Optional[str]:
    """
    Search host header files for a function or class declaration.

    Args:
        func_name: e.g., "distributed::MeshDevice::create_unit_mesh" or "CreateProgram"
        header_dirs: List of directories to search (typically tt-metalium/)

    Returns:
        The function signature(s) and relevant context, or None if not found
    """
    # Handle qualified names (namespace::class::method)
    parts = func_name.split("::")
    simple_name = parts[-1]  # Last part is the actual function/method name

    # Search patterns - handle different declaration styles
    patterns = [
        # static std::shared_ptr<MeshDevice> create_unit_mesh(...)
        rf"static\s+[\w:<>]+\s+{re.escape(simple_name)}\s*\([^)]*\)",
        # std::shared_ptr<MeshDevice> create_unit_mesh(...)
        rf"[\w:<>]+\s+{re.escape(simple_name)}\s*\([^)]*\)",
        # void CreateProgram(...)
        rf"\w+\s+{re.escape(simple_name)}\s*\([^)]*\)",
        # class CircularBufferConfig (for constructors)
        rf"class\s+{re.escape(simple_name)}\s*[{{:]",
        rf"struct\s+{re.escape(simple_name)}\s*[{{:]",
    ]

    results = []

    for header_dir in header_dirs:
        if not header_dir.exists():
            continue

        for header_file in header_dir.rglob("*.hpp"):
            try:
                content = header_file.read_text(encoding="utf-8", errors="ignore")

                # For qualified names, check if the namespace/class context matches
                if len(parts) > 1:
                    # Check if file contains the namespace/class
                    context_found = all(part in content for part in parts[:-1])
                    if not context_found:
                        continue

                for pattern in patterns:
                    matches = re.findall(pattern, content, re.MULTILINE)
                    if matches:
                        # Get surrounding context (e.g., comments above declaration)
                        for match in matches:
                            # Find position in file
                            idx = content.find(match)
                            if idx == -1:
                                continue

                            # Get ~10 lines before for documentation
                            lines_before = content[:idx].split("\n")[-10:]
                            func_line = match

                            # Clean up and format - collect doc comments
                            context_lines = []
                            for line in lines_before:
                                stripped = line.strip()
                                if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                                    context_lines.append(stripped)
                                elif not stripped:
                                    # Empty line - keep for spacing
                                    if context_lines:  # Only if we've started collecting
                                        context_lines.append("")

                            signature_block = "\n".join(context_lines + [func_line])
                            results.append(
                                {
                                    "function": func_name,
                                    "signature": func_line,
                                    "header": str(header_file.relative_to(TT_METAL_HOME)),
                                    "context": signature_block,
                                }
                            )
                        break  # Found in this file, move to next file
            except Exception as e:
                continue

    if not results:
        return None

    # Return formatted string with all signatures found
    formatted = f"// Function: {func_name}\n"
    for r in results[:1]:  # Just take first match to avoid bloat
        formatted += f"// Header: <{r['header']}>\n"
        formatted += f"{r['context']}\n\n"

    return formatted.strip()


def retrieve_host_api_signatures(host_examples: List[Dict[str, Any]], header_dirs: List[Path] = None) -> Dict[str, str]:
    """
    Main function: Retrieve host API signatures from example host code.

    Args:
        host_examples: List of host code example documents with 'chunk' field containing code
        header_dirs: Directories containing host API headers (tt-metalium/)

    Returns:
        Dict mapping function_name -> signature_with_context
    """
    if header_dirs is None:
        header_dirs = [
            TT_METAL_HOME / "tt_metal" / "include" / "tt-metalium",
        ]

    # Extract all host API calls from examples
    api_calls = set()
    for doc in host_examples:
        code = doc.get("chunk", "")
        api_calls.update(extract_host_api_calls_from_code(code))

    # Add common essential APIs that should always be included
    essential_apis = [
        "distributed::MeshDevice::create_unit_mesh",
        "distributed::MeshBuffer::create",
        "distributed::EnqueueWriteMeshBuffer",
        "distributed::EnqueueReadMeshBuffer",
        "distributed::EnqueueMeshWorkload",
        "distributed::Finish",
        "CreateProgram",
        "CreateKernel",
        "SetRuntimeArgs",
        "CreateCircularBuffer",
        "CircularBufferConfig",
        "TensorAccessorArgs",
    ]
    api_calls.update(essential_apis)

    # Retrieve signatures from headers
    signatures = {}
    for func_name in sorted(api_calls):
        sig = find_host_function_signature(func_name, header_dirs)
        if sig:
            signatures[func_name] = sig

    return signatures


def format_host_api_section(signatures: Dict[str, str]) -> str:
    """
    Format host API signatures into a prompt section.

    Returns a markdown section showing relevant host API functions.
    """
    if not signatures:
        return ""

    section = "## Host API Functions\n\n"
    section += "The following host APIs are available for device setup, buffer management, and kernel execution:\n\n"
    section += "```cpp\n"

    for func_name in sorted(signatures.keys()):
        section += signatures[func_name] + "\n\n"

    section += "```\n\n"

    return section


def create_host_template() -> str:
    """
    Create a canonical host code template showing correct structure and patterns.

    This template demonstrates:
    - Correct headers (<tt-metalium/...>)
    - Correct namespace usage (distributed::)
    - MeshDevice setup pattern
    - Circular buffer configuration
    - Kernel creation and runtime args
    - Program enqueue and execution
    """
    template = """
## Canonical Host Code Structure

Use this template as a guide for structuring your host code:

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// CORRECT HEADERS - use angle brackets for tt-metalium headers
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
    // 1. DEVICE SETUP - use distributed::MeshDevice for modern TT-Metal
    std::shared_ptr<distributed::MeshDevice> mesh_device =
        distributed::MeshDevice::create_unit_mesh(0);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range =
        distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    // 2. CORE RANGE SETUP
    constexpr CoreCoord core = {0, 0};

    // 3. INPUT DATA PREPARATION
    std::vector<bfloat16> input_vec(constants::TILE_HW);
    // ... fill input_vec with data ...

    // Tilize input data for device
    input_vec = tilize_nfaces(input_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // 4. DRAM BUFFER CREATION - use distributed::MeshBuffer
    constexpr uint32_t single_tile_size =
        sizeof(bfloat16) * constants::TILE_HEIGHT * constants::TILE_WIDTH;
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM
    };
    distributed::ReplicatedBufferConfig buffer_config{
        .size = sizeof(bfloat16) * input_vec.size()
    };
    std::shared_ptr<distributed::MeshBuffer> input_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    // Write input data to device
    distributed::EnqueueWriteMeshBuffer(cq, input_buffer, input_vec, false);

    // 5. CIRCULAR BUFFER SETUP
    constexpr uint32_t cb_index = CBIndex::c_0;
    CircularBufferConfig cb_config =
        CircularBufferConfig(single_tile_size, {{cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_config);

    // 6. KERNEL CREATION
    std::vector<uint32_t> compile_time_args = {cb_index};
    TensorAccessorArgs(*input_buffer).append_to(compile_time_args);

    KernelHandle kernel_id = CreateKernel(
        program,
        "path/to/kernel.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig{compile_time_args}
    );

    // 7. RUNTIME ARGS
    SetRuntimeArgs(program, kernel_id, core, {input_buffer->address()});

    // 8. PROGRAM EXECUTION
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // 9. READ RESULTS
    std::vector<bfloat16> result_vec(constants::TILE_HW);
    distributed::EnqueueReadMeshBuffer(cq, result_vec, output_buffer, true);

    // Untilize results
    result_vec = untilize_nfaces(result_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // 10. CLEANUP
    mesh_device->close();

    return 0;
}
```

KEY POINTS:
- Use `<tt-metalium/header.hpp>` NOT "tt_metal/header.hpp"
- Use `distributed::MeshDevice` NOT `Device`
- Use `distributed::MeshBuffer::create()` NOT `Buffer::create()`
- Use `distributed::EnqueueWriteMeshBuffer()` NOT `EnqueueWriteBuffer()`
- All distributed APIs are in the `distributed::` namespace
"""
    return template.strip()
