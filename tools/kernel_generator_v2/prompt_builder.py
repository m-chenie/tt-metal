from typing import List, Dict

from config import OPERATIONS, CORE_MODES
from api_retriever import retrieve_api_signatures, format_api_section


def build_system_prompt(op: str, core_mode: str, retrieved: List[Dict]) -> str:
    op_desc = OPERATIONS[op]["description"]
    op_cfg = OPERATIONS[op]
    core_desc = CORE_MODES[core_mode]["description"]
    op_type = op_cfg.get("operation_type", "binary")

    parts = [
        "You are an expert TT-Metal kernel developer for Tenstorrent hardware.",
        f"Target: {op} ({op_desc}), mode: {core_desc}.",
        "Follow patterns from the retrieved examples and respect core-mode specific dataflow/compute structure.",
        "",
    ]

    # Add SFPU-specific constraints
    if op_type == "sfpu_chain":
        parts.append("CRITICAL CONSTRAINTS FOR SFPU OPERATIONS:")
        parts.append("- ALL computation MUST use SFPU operations (element-wise on DST registers)")
        parts.append("- DO NOT create DRAM buffers for scalar constants - initialize them directly in circular buffers")
        parts.append("- Follow the sfpu_eltwise_chain pattern for constant initialization in reader kernel")

        # Add explicit input/constant specification if available
        variable_inputs = op_cfg.get("variable_inputs", [])
        constant_inputs = op_cfg.get("constant_inputs", {})
        cb_layout = op_cfg.get("circular_buffers", {})

        if variable_inputs or constant_inputs:
            parts.append("")
            parts.append("INPUT CONFIGURATION:")

            if variable_inputs:
                parts.append(f"- Variable inputs (from DRAM): {', '.join(variable_inputs)}")

            if constant_inputs:
                parts.append("- Constant inputs (initialize in reader kernel using float_to_bfloat16):")
                for const_name, const_val in constant_inputs.items():
                    parts.append(f"  * {const_name} = {const_val}")

            if cb_layout:
                parts.append("")
                parts.append("CIRCULAR BUFFER LAYOUT:")
                for cb_id, cb_desc in cb_layout.items():
                    parts.append(f"- {cb_id}: {cb_desc}")

        parts.append("")

    # Deduplicate by path (keep first occurrence)
    seen_paths = set()
    unique_docs = []
    for doc in retrieved:
        path = doc["path"]
        if path not in seen_paths:
            seen_paths.add(path)
            unique_docs.append(doc)

    # STAGE 1: Retrieve API signatures - use explicit required functions if specified
    formula = op_cfg.get("formula", "")
    math_steps = op_cfg.get("mathematical_steps", "")
    required_compute_functions = op_cfg.get("required_compute_functions", None)

    # For SFPU operations, use focused header directories
    if op_type == "sfpu_chain":
        from config import SFPU_HEADER_DIRS

        header_dirs = SFPU_HEADER_DIRS
    else:
        header_dirs = None  # Use default API_HEADER_DIRS

    api_signatures = retrieve_api_signatures(
        retrieved_examples=unique_docs,
        operation_description=op_desc,
        formula=formula,
        math_steps=math_steps,
        header_dirs=header_dirs,
        required_compute_functions=required_compute_functions,
    )

    # Add API signatures section BEFORE examples
    if api_signatures:
        api_section = format_api_section(api_signatures)
        parts.append(api_section)

    # Group by kernel type for better organization
    compute_docs = [d for d in unique_docs if d["meta"].get("kind") == "compute"]
    reader_docs = [d for d in unique_docs if d["meta"].get("kind") == "reader"]
    writer_docs = [d for d in unique_docs if d["meta"].get("kind") == "writer"]
    other_docs = [d for d in unique_docs if d not in compute_docs + reader_docs + writer_docs]

    # Add compute examples
    if compute_docs:
        parts.append("## Compute Kernel Examples")
        for doc in compute_docs[:2]:  # Limit to avoid bloat
            parts.append(f"# Source: {doc['path']}\n```cpp\n{doc['chunk']}\n```\n")

    # Add reader examples
    if reader_docs:
        parts.append("## Reader Kernel Examples")
        for doc in reader_docs[:1]:  # Just one example
            parts.append(f"# Source: {doc['path']}\n```cpp\n{doc['chunk']}\n```\n")

    # Add writer examples
    if writer_docs:
        parts.append("## Writer Kernel Examples")
        for doc in writer_docs[:1]:  # Just one example
            parts.append(f"# Source: {doc['path']}\n```cpp\n{doc['chunk']}\n```\n")

    parts.append("Generate concise, correct TT-Metal code following these patterns.")
    return "\n\n".join(parts)


def build_kernel_user_prompt(op: str, core_mode: str) -> str:
    op_desc = OPERATIONS[op]["description"]
    op_cfg = OPERATIONS[op]
    compute_name = op_cfg["compute_kernel"]
    op_type = op_cfg.get("operation_type", "binary")

    if core_mode == "single":
        reader_name = "reader_binary_1_tile.cpp"
        writer_name = "writer_1_tile.cpp"
    else:
        reader_name = "reader_binary_tiles_partitioned.cpp"
        writer_name = "writer_tiles_partitioned.cpp"

    # Build requirements based on operation type WITHOUT prescribing specific API calls
    if op_type == "sfpu_chain":
        # Get formula and config details
        formula = op_cfg.get("formula", op_desc)
        math_steps = op_cfg.get("mathematical_steps", "")
        variable_inputs = op_cfg.get("variable_inputs", [])
        constant_inputs = op_cfg.get("constant_inputs", {})
        cb_layout = op_cfg.get("circular_buffers", {})

        # Build circular buffer description from config
        if cb_layout:
            cb_lines = ["Circular buffer layout (MUST follow exactly):"]
            for cb_id, cb_desc_text in cb_layout.items():
                cb_lines.append(f"  * {cb_id}: {cb_desc_text}")
            cb_desc = "\n".join(cb_lines)
        else:
            cb_desc = "Use standard circular buffer layout: CB_0/CB_1 for inputs, CB_16 for output"

        # Build reader description
        if variable_inputs and constant_inputs:
            reader_parts = [f"Read {variable_inputs[0]} from DRAM into CB_0."]
            reader_parts.append(f"Initialize constant tiles in reader kernel using float_to_bfloat16 pattern:")
            for const_name, const_val in constant_inputs.items():
                reader_parts.append(f"  * {const_name} = {const_val} (in appropriate CB as specified above)")
            reader_desc = " ".join(reader_parts)
        else:
            reader_desc = "Read input tiles from DRAM using NOC async operations"

        compute_desc = f"Implement the formula: {formula}. Mathematical steps: {math_steps}. DO NOT initialize constants in compute kernel."

        requirements = f"""- {cb_desc}
- Compute kernel: {compute_desc}. Use appropriate SFPU operations from the examples. Follow the pattern: initialize operations, wait for inputs, acquire registers, perform computation, pack result, release registers
- Reader kernel: {reader_desc}. Use noc_async_read with barriers
- Writer kernel: Write output tiles from CB_16 to DRAM using noc_async_write with barriers"""
    else:
        # Binary operations
        requirements = """- Use circular buffers: CB_0 and CB_1 for inputs, CB_16 for output
- Compute kernel: Perform the operation on input tiles using appropriate compute APIs
- Reader kernel: Read input tiles from DRAM to circular buffers
- Writer kernel: Write result tiles from circular buffer to DRAM"""

    return f"""Generate TT-Metal kernels for {op_desc} ({core_mode}-core).

Requirements:
- Emit exactly three separate code blocks in your response
- Label each block clearly: COMPUTE, READER, WRITER
{requirements}
- Study the provided examples to identify the correct API functions and usage patterns
- Follow TT-Metal conventions: cb_wait/reserve/push/pop discipline, NOC barriers, proper includes

Expected output format:
```cpp
// COMPUTE KERNEL: {compute_name}
[compute kernel code here]
```

```cpp
// READER KERNEL: {reader_name}
[reader kernel code here]
```

```cpp
// WRITER KERNEL: {writer_name}
[writer kernel code here]
```"""


def build_host_system_prompt(op: str, core_mode: str, host_examples: List[Dict]) -> str:
    """
    Build system prompt specifically for host code generation.

    Includes:
    2. Complete host code examples
    3. Host API signatures extracted from examples
    """
    from host_api_retriever import retrieve_host_api_signatures, format_host_api_section

    op_desc = OPERATIONS[op]["description"]
    op_cfg = OPERATIONS[op]
    core_desc = CORE_MODES[core_mode]["description"]

    parts = [
        "You are an expert TT-Metal host code developer for Tenstorrent hardware.",
        f"Target: {op} ({op_desc}), mode: {core_desc}.",
        "Generate correct, modern TT-Metal host code following the canonical structure and examples.",
        "",
    ]

    # Retrieve and add host API signatures
    api_signatures = retrieve_host_api_signatures(host_examples)
    if api_signatures:
        api_section = format_host_api_section(api_signatures)
        parts.append(api_section)

    # Add complete host code examples
    if host_examples:
        parts.append("## Complete Host Code Examples")
        parts.append("Study these examples to understand the full workflow:\n")

        for doc in host_examples[:2]:  # Limit to 2 complete examples
            parts.append(f"### Example: {doc['path']}\n")
            parts.append(f"```cpp\n{doc['chunk']}\n```\n")

    # Add CMakeLists.txt examples
    from retriever import retrieve_cmake_examples

    cmake_examples = retrieve_cmake_examples(op)
    if cmake_examples:
        parts.append("## CMakeLists.txt Examples")
        parts.append("Use these patterns for proper library linking:\n")

        for doc in cmake_examples[:2]:  # Limit to 2 examples
            parts.append(f"### Example: {doc['path']}\n")
            parts.append(f"```cmake\n{doc['chunk']}\n```\n")

    parts.append("CRITICAL REQUIREMENTS:")
    parts.append("- Use ONLY angle bracket includes: `#include <tt-metalium/host_api.hpp>` NOT quotes")
    parts.append("- Use `distributed::MeshDevice` NOT `Device`")
    parts.append("- Use `distributed::MeshBuffer` NOT `Buffer`")
    parts.append("- All distributed APIs require `distributed::` namespace prefix")
    parts.append("- CMakeLists.txt MUST include: find_package(TT-Metalium) and target_link_libraries(...TT::Metalium)")

    return "\n\n".join(parts)


def build_host_user_prompt(op: str, core_mode: str) -> str:
    op_desc = OPERATIONS[op]["description"]
    op_type = OPERATIONS[op].get("operation_type", "binary")

    # Build CB configuration hint based on operation type
    if op_type == "sfpu_chain":
        op_cfg = OPERATIONS[op]
        inputs = op_cfg.get("inputs", [])
        constants = op_cfg.get("constants", [])

        if inputs and constants:
            cb_hint = f"Configure CB_0 ({inputs[0]}), CB_1 ({inputs[1]}), CB_2 ({constants[0]} constant), CB_16 (output). Initialize CB_2 with the constant {constants[0]} value."
        else:
            cb_hint = "Configure CB_0, CB_1 for inputs, CB_16 for output."
    else:
        cb_hint = "Configure circular buffers: CB_0, CB_1 for inputs; CB_16 for output."

    return f"""Generate complete host code (.cpp file) AND CMakeLists.txt for {op_desc} ({core_mode}-core).

Requirements:
- Use correct headers: `#include <tt-metalium/host_api.hpp>` with angle brackets
- Use `distributed::MeshDevice::create_unit_mesh()` for device setup
- Create DRAM buffers using `distributed::MeshBuffer::create()`
- {cb_hint}
- Compile and launch the three kernels (reader, compute, writer) with SetRuntimeArgs
- Enqueue program using `distributed::EnqueueMeshWorkload()`
- Add CPU golden validation with PCC check
- Use proper tilize/untilize for data conversion

Output format - provide TWO code blocks:

```cpp
// HOST CODE: {op}_{core_mode}_v2.cpp
[complete host code here]
```

```cmake
# CMakeLists.txt
[complete CMakeLists.txt with proper linking]
```

Ensure CMakeLists.txt includes find_package(TT-Metalium) and target_link_libraries with TT::Metalium."""
