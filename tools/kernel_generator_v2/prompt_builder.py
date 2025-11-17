from typing import List, Dict

from config import OPERATIONS, CORE_MODES
from api_retriever import retrieve_api_signatures, format_api_section


def build_system_prompt(op: str, core_mode: str, retrieved: List[Dict]) -> str:
    op_desc = OPERATIONS[op]["description"]
    op_cfg = OPERATIONS[op]
    core_desc = CORE_MODES[core_mode]["description"]
    parts = [
        "You are an expert TT-Metal kernel developer for Tenstorrent hardware.",
        f"Target: {op} ({op_desc}), mode: {core_desc}.",
        "Follow patterns from the retrieved examples and respect core-mode specific dataflow/compute structure.",
        "",
    ]

    # Deduplicate by path (keep first occurrence)
    seen_paths = set()
    unique_docs = []
    for doc in retrieved:
        path = doc["path"]
        if path not in seen_paths:
            seen_paths.add(path)
            unique_docs.append(doc)

    # STAGE 1: Retrieve API signatures from examples + operation requirements
    formula = op_cfg.get("formula", "")
    math_steps = op_cfg.get("mathematical_steps", "")

    api_signatures = retrieve_api_signatures(
        retrieved_examples=unique_docs, operation_description=op_desc, formula=formula, math_steps=math_steps
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
        # Get formula and inputs from config
        formula = op_cfg.get("formula", op_desc)
        math_steps = op_cfg.get("mathematical_steps", "")
        inputs = op_cfg.get("inputs", [])
        constants = op_cfg.get("constants", [])

        # Describe WHAT to compute, not HOW
        if inputs and constants:
            cb_desc = f"Circular buffers: CB_0 for {inputs[0]}, CB_1 for {inputs[1]}, CB_2 for constant {constants[0]}, CB_16 for output"
            compute_desc = f"Implement the formula: {formula}. Mathematical steps: {math_steps}"
            reader_desc = f"Read {inputs[0]} and {inputs[1]} tiles from DRAM. Initialize CB_2 with the constant {constants[0]} value"
        else:
            cb_desc = "Use standard circular buffer layout: CB_0/CB_1 for inputs, CB_16 for output"
            compute_desc = f"Implement the operation: {formula}"
            reader_desc = "Read input tiles from DRAM using NOC async operations"

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


def build_host_user_prompt(op: str, core_mode: str) -> str:
    op_desc = OPERATIONS[op]["description"]
    op_type = OPERATIONS[op].get("operation_type", "binary")
    if op_type == "sfpu_chain":
        cb_hint = "Configure CB_0 (V), CB_1 (vj), CB_2 (isat constant), CB_16 (output). Load a single isat tile into CB_2 once."
    else:
        cb_hint = "Configure circular buffers for kernels: CB_0, CB_1 inputs; CB_16 output."

    return f"""Generate host code (single .cpp) for {op_desc} ({core_mode}-core).

Requirements:
- Include TT-Metal host headers and use distributed::MeshDevice mesh setup.
- Create DRAM buffers for inputs/outputs.
- {cb_hint}
- Compile and launch the three kernels (reader, compute, writer), set runtime args, and enqueue program.
- Add simple validation against a CPU golden.
Return only the code."""
