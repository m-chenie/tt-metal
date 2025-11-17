"""
API function signature retrieval.

Two-stage retrieval:
1. Extract API functions from retrieved example code
2. Map user prompt operations to API function names
3. Retrieve actual function signatures from header files
"""

import re
from pathlib import Path
from typing import List, Dict, Set, Any, Optional
from config import TT_METAL_HOME


# Mapping from operation keywords to API function patterns
OPERATION_TO_API = {
    # Arithmetic operations
    "divide": ["div_binary_tile", "div_tiles"],
    "multiply": ["mul_binary_tile", "mul_tiles"],
    "subtract": ["sub_binary_tile", "sub_tiles"],
    "add": ["add_binary_tile", "add_tiles"],
    # Transcendental/special functions
    "exponent": ["exp_tile"],
    "exponential": ["exp_tile"],
    "exp": ["exp_tile"],
    "logarithm": ["log_tile"],
    "log": ["log_tile"],
    "power": ["power_binary_tile", "power_tile"],
    "sqrt": ["sqrt_tile"],
    "square_root": ["sqrt_tile"],
    # Trigonometric
    "sine": ["sin_tile"],
    "sin": ["sin_tile"],
    "cosine": ["cos_tile"],
    "cos": ["cos_tile"],
    "tangent": ["tan_tile"],
    "tan": ["tan_tile"],
    # Activation functions
    "relu": ["relu_tile"],
    "sigmoid": ["sigmoid_tile"],
    "tanh": ["tanh_tile"],
    "gelu": ["gelu_tile"],
    "softmax": ["softmax_tile"],
}


def extract_api_calls_from_code(code: str) -> Set[str]:
    """
    Parse C++ code and extract all API function calls.

    Looks for patterns like:
    - exp_tile(...)
    - add_binary_tile(...)
    - noc_async_read_tile(...)
    - init_sfpu(...)

    Returns set of function names (without arguments).
    """
    # Pattern: word characters followed by underscore/letters and then opening paren
    # Common TT-Metal API patterns: *_tile, *_init, noc_*, cb_*
    pattern = r"\b([a-z_][a-z0-9_]*(?:_tile|_init|_barrier|_cb|_sfpu|noc_\w+|cb_\w+))\s*\("

    matches = re.findall(pattern, code, re.IGNORECASE)

    # Also catch standalone function calls that might not match the pattern
    # This catches things like "copy_tile", "pack_tile", etc.
    general_pattern = r"\b([a-z_][a-z0-9_]*)\s*\("
    general_matches = re.findall(general_pattern, code, re.IGNORECASE)

    # Filter general matches to likely API calls (avoid C++ keywords)
    cpp_keywords = {"if", "for", "while", "switch", "return", "sizeof", "static_cast", "reinterpret_cast"}
    api_calls = set(matches)

    for func in general_matches:
        if func not in cpp_keywords and ("_" in func or func.startswith("noc") or func.startswith("cb")):
            api_calls.add(func)

    return api_calls


def map_operations_to_apis(operation_description: str, formula: str = "", math_steps: str = "") -> Set[str]:
    """
    Map high-level operation descriptions to specific API function names.

    Args:
        operation_description: e.g., "diode current equation (I = isat × (exp(V/vj) - 1))"
        formula: e.g., "I = isat × (exp(V/vj) - 1)"
        math_steps: e.g., "divide V by vj, exponentiate result, subtract 1, multiply by isat"

    Returns:
        Set of API function names like {"div_binary_tile", "exp_tile", "sub_binary_tile", "mul_binary_tile"}
    """
    apis = set()

    # Combine all text sources
    text = f"{operation_description} {formula} {math_steps}".lower()

    # Check each operation keyword
    for keyword, api_functions in OPERATION_TO_API.items():
        if keyword in text:
            apis.update(api_functions)

    # Special handling for common mathematical operations
    if "÷" in text or "/" in text or "divide" in text:
        apis.add("div_binary_tile")
    if "×" in text or "*" in text or "multiply" in text:
        apis.add("mul_binary_tile")
    if "−" in text or "-" in text or "subtract" in text:
        apis.add("sub_binary_tile")
    if "+" in text or "add" in text:
        apis.add("add_binary_tile")

    return apis


def find_function_signature(func_name: str, header_dirs: List[Path]) -> Optional[str]:
    """
    Search header files for a function signature.

    Args:
        func_name: e.g., "div_binary_tile"
        header_dirs: List of directories to search

    Returns:
        The function signature(s) and relevant context, or None if not found
    """
    # Search patterns - handle different declaration styles
    patterns = [
        # ALWI void div_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)
        rf"ALWI\s+\w+\s+{re.escape(func_name)}\s*\([^)]*\)",
        # inline void div_binary_tile(...)
        rf"inline\s+\w+\s+{re.escape(func_name)}\s*\([^)]*\)",
        # void div_binary_tile(...)
        rf"\b\w+\s+{re.escape(func_name)}\s*\([^)]*\)",
    ]

    results = []

    for header_dir in header_dirs:
        if not header_dir.exists():
            continue

        for header_file in header_dir.rglob("*.h"):
            try:
                content = header_file.read_text(encoding="utf-8", errors="ignore")

                for pattern in patterns:
                    matches = re.findall(pattern, content, re.MULTILINE)
                    if matches:
                        # Get surrounding context (e.g., comments above function)
                        for match in matches:
                            # Find position in file
                            idx = content.find(match)
                            if idx == -1:
                                continue

                            # Get ~5 lines before for documentation
                            lines_before = content[:idx].split("\n")[-5:]
                            func_line = match

                            # Clean up and format
                            context_lines = []
                            for line in lines_before:
                                line = line.strip()
                                if line.startswith("//") or line.startswith("/*") or line.startswith("*"):
                                    context_lines.append(line)

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
    for r in results:
        formatted += f"// Header: {r['header']}\n"
        formatted += f"{r['context']}\n\n"

    return formatted.strip()


def retrieve_api_signatures(
    retrieved_examples: List[Dict[str, Any]],
    operation_description: str = "",
    formula: str = "",
    math_steps: str = "",
    header_dirs: List[Path] = None,
) -> Dict[str, str]:
    """
    Main function: Retrieve API signatures from both examples and operation requirements.

    Args:
        retrieved_examples: List of example documents with 'chunk' field containing code
        operation_description: Description of the operation to implement
        formula: Mathematical formula
        math_steps: Step-by-step mathematical operations
        header_dirs: Directories containing API headers

    Returns:
        Dict mapping function_name -> signature_with_context
    """
    if header_dirs is None:
        from config import API_HEADER_DIRS

        header_dirs = API_HEADER_DIRS

    # Stage 1: Extract APIs from retrieved examples
    api_calls_from_examples = set()
    for doc in retrieved_examples:
        code = doc.get("chunk", "")
        api_calls_from_examples.update(extract_api_calls_from_code(code))

    # Stage 2: Map operation requirements to APIs
    api_calls_from_prompt = map_operations_to_apis(operation_description, formula, math_steps)

    # Combine both sets
    all_api_calls = api_calls_from_examples | api_calls_from_prompt

    # Stage 3: Retrieve signatures from headers
    signatures = {}
    for func_name in sorted(all_api_calls):
        sig = find_function_signature(func_name, header_dirs)
        if sig:
            signatures[func_name] = sig

    return signatures


def format_api_section(signatures: Dict[str, str]) -> str:
    """
    Format API signatures into a prompt section.

    Returns a markdown section showing relevant API functions.
    """
    if not signatures:
        return ""

    section = "## Relevant API Functions\n\n"
    section += "```cpp\n"

    for func_name in sorted(signatures.keys()):
        section += signatures[func_name] + "\n\n"

    section += "```\n\n"

    return section
