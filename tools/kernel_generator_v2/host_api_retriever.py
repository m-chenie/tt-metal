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
