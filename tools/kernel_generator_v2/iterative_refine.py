"""
Iterative refinement: compile generated code, feed errors back to LLM, regenerate.
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from groq import Groq

from config import TT_METAL_HOME, MODEL_DEFAULT, TEMPERATURE, MAX_TOKENS
from host_api_retriever import retrieve_host_api_signatures, create_host_template
from retriever import retrieve_host_examples

logger = logging.getLogger("kernel_generator_v2")


def compile_example(example_path: Path) -> Dict[str, any]:
    """
    Compile programming examples using build_metal.sh --build-programming-examples.

    Args:
        example_path: Path to the example directory (e.g., diode_equation_single_core_v2_generated)

    Returns:
        {
            'success': bool,
            'stdout': str,
            'stderr': str,
            'exit_code': int,
            'error_summary': str
        }
    """
    logger.info(f"Compiling: {example_path.name}")

    # Build all programming examples using TT-Metal build system
    cmd = [str(TT_METAL_HOME / "build_metal.sh"), "--build-programming-examples"]

    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=TT_METAL_HOME, capture_output=True, text=True, timeout=300)  # 5 minute timeout

        # Combine stdout and stderr
        combined_output = result.stdout + "\n" + result.stderr

        # Check if our specific example had errors
        example_name = example_path.name
        example_mentioned = example_name in combined_output

        if example_mentioned and ("error:" in combined_output.lower() or "undefined" in combined_output.lower()):
            # Our example had errors
            success = False
        else:
            # Either our example wasn't mentioned (not built) or built successfully
            success = result.returncode == 0

        error_summary = parse_compilation_errors(combined_output) if not success else ""

        return {
            "success": success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "combined": combined_output,
            "exit_code": result.returncode,
            "error_summary": error_summary,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Compilation timeout (>5 minutes)",
            "combined": "Compilation timeout (>5 minutes)",
            "exit_code": -1,
            "error_summary": "Compilation took too long - likely infinite loop or hang",
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "combined": str(e),
            "exit_code": -1,
            "error_summary": f"Failed to run build command: {e}",
        }


def parse_compilation_errors(output: str) -> str:
    """
    Parse and categorize compilation errors into a concise summary.

    Extracts:
    - Linker errors (undefined reference)
    - Syntax errors
    - Missing include errors
    - API signature mismatches

    Args:
        output: Combined stdout + stderr from build

    Returns:
        Concise error summary (not full output)
    """
    if not output or not output.strip():
        return "No error output captured"

    errors = {"linker": [], "syntax": [], "includes": [], "api": [], "cmake": [], "other": []}

    lines = output.split("\n")

    for line in lines:
        lower_line = line.lower()

        # Linker errors
        if "undefined reference" in lower_line:
            # Extract the symbol name
            match = re.search(r"undefined reference to [`']([^'`]+)", line)
            if match:
                symbol = match.group(1)
                # Simplify symbol (remove template/namespace cruft)
                simple_symbol = symbol.split("(")[0].split("<")[0]
                if simple_symbol not in errors["linker"]:
                    errors["linker"].append(simple_symbol)

        # Include errors
        elif "fatal error:" in lower_line and (".h" in lower_line or ".hpp" in lower_line):
            match = re.search(r"fatal error: ([^:]+): No such file", line)
            if match:
                errors["includes"].append(match.group(1))

        # Syntax/compilation errors
        elif "error:" in lower_line and ".cpp:" in lower_line:
            # Extract file:line: error message
            match = re.search(r"([^/\s]+\.cpp):(\d+):\d+: error: (.+)", line)
            if match:
                file, lineno, msg = match.groups()
                errors["syntax"].append(f"{file}:{lineno}: {msg[:100]}")

        # API signature mismatch
        elif "no matching function" in lower_line or "cannot convert" in lower_line:
            if line.strip() and "error:" in lower_line:
                errors["api"].append(line.strip()[:150])

        # CMake errors
        elif "cmake error" in lower_line:
            if line.strip():
                errors["cmake"].append(line.strip()[:200])

    # Build summary
    summary_parts = []

    if errors["linker"]:
        unique_linker = list(set(errors["linker"]))[:10]  # Top 10 unique symbols
        summary_parts.append(f"LINKER ERRORS ({len(unique_linker)} unique symbols):")
        for symbol in unique_linker:
            summary_parts.append(f"  - undefined reference to: {symbol}")

    if errors["includes"]:
        unique_includes = list(set(errors["includes"]))
        summary_parts.append(f"\nINCLUDE ERRORS:")
        for inc in unique_includes:
            summary_parts.append(f"  - missing header: {inc}")

    if errors["syntax"]:
        summary_parts.append(f"\nCOMPILATION ERRORS ({len(errors['syntax'])} errors):")
        for err in errors["syntax"][:5]:  # First 5
            summary_parts.append(f"  - {err}")

    if errors["api"]:
        summary_parts.append(f"\nAPI SIGNATURE ERRORS:")
        for err in errors["api"][:3]:  # First 3
            summary_parts.append(f"  - {err}")

    if errors["cmake"]:
        summary_parts.append(f"\nCMAKE ERRORS:")
        for err in errors["cmake"][:5]:  # First 5
            summary_parts.append(f"  - {err}")

    if not any(errors.values()):
        # No categorized errors, show last 20 lines of stderr
        summary_parts.append("BUILD FAILED - Last 20 lines of error output:")
        summary_parts.extend(lines[-20:])

    return "\n".join(summary_parts)


def detect_error_type(error_summary: str) -> str:
    """
    Determine primary error type to guide regeneration strategy.

    Returns: 'cmake', 'linker', 'syntax', 'includes', 'api', or 'unknown'
    """
    if "CMAKE ERRORS" in error_summary or "find_package" in error_summary.lower():
        return "cmake"
    elif "LINKER ERRORS" in error_summary:
        return "linker"
    elif "INCLUDE ERRORS" in error_summary:
        return "includes"
    elif "API SIGNATURE" in error_summary:
        return "api"
    elif "COMPILATION ERRORS" in error_summary:
        return "syntax"
    else:
        return "unknown"


def build_refinement_prompt(
    current_files: Dict[str, str], error_summary: str, error_type: str, iteration: int, example_path: Path = None
) -> Tuple[str, str]:
    """
    Build system and user prompts for fixing compilation errors.

    Args:
        current_files: Dict with keys like 'compute', 'reader', 'writer', 'host_code', 'cmake'
        error_summary: Parsed error summary
        error_type: Primary error type (linker/syntax/etc)
        iteration: Current iteration number
        example_path: Path to example (used to infer operation name)

    Returns:
        (system_prompt, user_prompt)
    """
    # Get host API context
    host_template = create_host_template()

    # Infer operation from example path if possible, otherwise use generic "binary"
    operation = "binary"  # default fallback
    if example_path:
        # Extract operation name from path like "diode_equation_single_core_v2_generated"
        path_name = example_path.name
        # Try to match known operations from the name
        for known_op in ["add", "sub", "mul", "div", "exp", "log", "sqrt", "diode_equation"]:
            if known_op in path_name.lower():
                operation = known_op
                break

    host_examples = retrieve_host_examples(operation)

    # Format host examples for display
    host_examples_text = ""
    for i, example in enumerate(host_examples[:2], 1):  # Show top 2 examples
        host_examples_text += f"\n### Example {i} ({example.get('path', 'unknown')})\n"
        host_examples_text += f"```cpp\n{example.get('chunk', '')[:2000]}\n```\n"

    host_api_signatures = retrieve_host_api_signatures(host_examples)

    system_prompt = f"""You are an expert TT-Metal developer fixing compilation errors.

This is iteration {iteration} of debugging generated code that failed to compile.

CANONICAL HOST CODE TEMPLATE:
```cpp
{host_template}
```

HOST API EXAMPLES:
{host_examples_text}

HOST API SIGNATURES:
{host_api_signatures}

CURRENT CODE:
"""

    # Always show ALL code files regardless of error type
    # This provides complete context to the LLM

    # Show kernels first
    for key in ["compute", "reader", "writer"]:
        if key in current_files:
            system_prompt += f"\n## {key.title()} Kernel\n"
            system_prompt += f"```cpp\n{current_files[key]}\n```\n\n"

    # Show host code
    if "host_code" in current_files:
        system_prompt += "\n## Host Code\n"
        system_prompt += f"```cpp\n{current_files['host_code']}\n```\n\n"

    # Show CMakeLists.txt
    system_prompt += "\n## CMakeLists.txt\n"
    system_prompt += f"```cmake\n{current_files.get('cmake', 'N/A')}\n```\n\n"

    system_prompt += (
        """
COMPILATION ERRORS:
```
"""
        + error_summary
        + """
```

Your task: Fix the errors and provide corrected code.
"""
    )

    # User prompt varies by error type
    if error_type == "cmake":
        user_prompt = """CMake configuration error.

FIX REQUIRED:
1. Replace the CMakeLists.txt with correct content
2. MUST include: find_package(TT-Metalium REQUIRED)
3. MUST include: target_link_libraries(...PROJECT_NAME... PUBLIC TT::Metalium)
4. Do NOT use find_package(OpenMPI) or any MPI-related commands
5. Follow the canonical CMakeLists.txt pattern

CRITICAL: Provide COMPLETE, FULL code files - not just the changed sections.
Include ALL includes, ALL functions, ALL code in each file.
Provide: compute kernel (full), reader kernel (full), writer kernel (full), host code (full), cmake (full).
"""

    elif error_type == "linker":
        user_prompt = """The code compiles but fails at link time due to missing library references.

FIX REQUIRED:
1. Update CMakeLists.txt to link against required libraries
2. Ensure target_link_libraries includes TT::Metalium
3. Check if find_package(TT-Metalium REQUIRED) is present

CRITICAL: Provide COMPLETE, FULL code files - not just the changed sections.
Include ALL includes, ALL functions, ALL code in each file.
Provide: compute kernel (full), reader kernel (full), writer kernel (full), host code (full), cmake (full).
"""

    elif error_type == "includes":
        user_prompt = """Missing header files.

FIX REQUIRED:
1. Add correct #include statements using angle brackets: <tt-metalium/...>
2. Do NOT use quotes for TT-Metal headers
3. Ensure all includes exist in the codebase

CRITICAL: Provide COMPLETE, FULL code files - not just the changed sections.
Include ALL includes, ALL functions, ALL code in each file.
Provide: compute kernel (full), reader kernel (full), writer kernel (full), host code (full), cmake (full).
"""

    elif error_type == "syntax":
        user_prompt = f"""Compilation errors in the code.

ERROR TYPE: {error_type}

FIX REQUIRED:
1. Fix syntax errors or API signature mismatches
2. Use correct TT-Metal API functions (check the error messages)
3. Ensure proper namespace usage (distributed::, tt::, etc)

CRITICAL: Provide COMPLETE, FULL code files - not just the changed sections.
Include ALL includes, ALL functions, ALL code in each file.
Provide: compute kernel (full), reader kernel (full), writer kernel (full), host code (full), cmake (full).
"""

    elif error_type == "api":
        user_prompt = """API usage errors detected.

FIX REQUIRED:
1. Fix incorrect API function calls
2. Match function signatures correctly
3. Use correct argument types

CRITICAL: Provide COMPLETE, FULL code files - not just the changed sections.
Include ALL includes, ALL functions, ALL code in each file.
Provide: compute kernel (full), reader kernel (full), writer kernel (full), host code (full), cmake (full).
"""

    else:  # unknown
        user_prompt = """Unknown compilation failure.

Analyze the error output and fix the issues.

CRITICAL: Provide COMPLETE, FULL code files - not just the changed sections.
Include ALL includes, ALL functions, ALL code in each file.
Provide: compute kernel (full), reader kernel (full), writer kernel (full), host code (full), cmake (full).
"""

    return system_prompt, user_prompt


def iterative_refine(client: Groq, args):
    """
    Main iterative refinement loop.

    Args:
        client: Groq API client
        args: Parsed command-line arguments
    """
    # Pre-flight check: verify API key is set
    if not client.api_key or client.api_key == "":
        raise SystemExit(
            "ERROR: Groq API key not set. Please set GROQ_API_KEY environment variable.\n"
            "Example: export GROQ_API_KEY='your-api-key-here'"
        )

    example_path = Path(args.example_path)

    if not example_path.exists():
        raise SystemExit(f"Example path does not exist: {example_path}")

    logger.info(f"Starting iterative refinement on: {example_path}")
    logger.info(f"Max iterations: {args.max_iterations}")

    # Initial compilation attempt
    for iteration in range(1, args.max_iterations + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}/{args.max_iterations}")
        logger.info(f"{'='*60}\n")

        # Compile
        result = compile_example(example_path)

        if result["success"]:
            logger.info("✓ ✓ ✓ COMPILATION SUCCESSFUL! ✓ ✓ ✓")
            logger.info(f"Example compiled successfully after {iteration} iteration(s)")
            return

        logger.warning(f"✗ Compilation failed (exit code: {result['exit_code']})")

        # Debug: Save full output to file
        debug_file = example_path / f"build_output_iteration_{iteration}.txt"
        debug_file.write_text(
            f"=== STDOUT ===\n{result['stdout']}\n\n=== STDERR ===\n{result['stderr']}\n\n=== COMBINED ===\n{result.get('combined', '')}"
        )
        logger.info(f"Saved full build output to: {debug_file}")

        logger.info("\nError Summary:")
        logger.info(result["error_summary"])

        # Detect error type
        error_type = detect_error_type(result["error_summary"])
        logger.info(f"\nDetected error type: {error_type}")

        # Load current files
        current_files = load_current_files(example_path)

        # Build refinement prompt
        system_prompt, user_prompt = build_refinement_prompt(
            current_files, result["error_summary"], error_type, iteration, example_path
        )

        if args.save_prompt:
            prompt_file = example_path / f"refinement_iteration_{iteration}.md"
            prompt_file.write_text(
                f"# Iteration {iteration}\n\n## System\n```\n{system_prompt}\n```\n\n## User\n```\n{user_prompt}\n```\n"
            )
            logger.info(f"Saved refinement prompt to: {prompt_file}")

        # Call LLM to fix errors
        logger.info("Calling LLM for fixes...")
        try:
            resp = client.chat.completions.create(
                model=args.model or MODEL_DEFAULT,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            content = resp.choices[0].message.content

            # Save LLM response to debug file
            if args.save_prompt:
                prompt_file = example_path / f"refinement_iteration_{iteration}.md"
                # Append the response to the existing file
                with open(prompt_file, "a") as f:
                    f.write(f"\n\n## LLM Response\n```\n{content}\n```\n")
                logger.info(f"Saved LLM response to: {prompt_file}")

            # Apply fixes
            apply_fixes(example_path, content, error_type, current_files)
            logger.info("Applied fixes from LLM")

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            logger.error("Cannot continue iteration")
            return

    # Max iterations reached
    logger.error(f"\n✗ Failed to compile after {args.max_iterations} iterations")
    logger.error("Manual intervention may be required")


def load_current_files(example_path: Path) -> Dict[str, str]:
    """Load all current code files from the example directory."""
    files = {}

    # Load kernels
    kernels_dir = example_path / "kernels"
    if (kernels_dir / "compute").exists():
        for cpp_file in (kernels_dir / "compute").glob("*.cpp"):
            files["compute"] = cpp_file.read_text()
            break

    if (kernels_dir / "dataflow").exists():
        for cpp_file in (kernels_dir / "dataflow").glob("*reader*.cpp"):
            files["reader"] = cpp_file.read_text()
            break
        for cpp_file in (kernels_dir / "dataflow").glob("*writer*.cpp"):
            files["writer"] = cpp_file.read_text()
            break

    # Load host code
    for cpp_file in example_path.glob("*.cpp"):
        files["host_code"] = cpp_file.read_text()
        break

    # Load CMakeLists.txt
    cmake_file = example_path / "CMakeLists.txt"
    if cmake_file.exists():
        files["cmake"] = cmake_file.read_text()

    return files


def apply_fixes(example_path: Path, llm_response: str, error_type: str, current_files: Dict[str, str]):
    """
    Apply fixes from LLM response to the example files.

    Extracts code blocks and overwrites the appropriate files.
    """
    if "```" not in llm_response:
        logger.warning("No code blocks found in LLM response")
        logger.warning(f"LLM response preview: {llm_response[:500]}")
        return

    # Extract code blocks
    blocks = llm_response.split("```")
    logger.info(f"Found {len(blocks) // 2} code blocks in LLM response")

    files_updated = 0
    for i in range(1, len(blocks), 2):
        block = blocks[i]
        lines = block.split("\n", 1)
        if len(lines) > 1:
            lang_tag = lines[0].strip().lower()
            code = lines[1].strip()
        else:
            lang_tag = ""
            code = block.strip()

        # Determine what to update based on lang tag and content
        if "cmake" in lang_tag or "cmake_minimum_required" in code[:100].lower():
            # Update CMakeLists.txt
            cmake_file = example_path / "CMakeLists.txt"
            cmake_file.write_text(code)
            logger.info("Updated CMakeLists.txt")
            files_updated += 1

        elif "compute" in lang_tag.lower() or "// compute kernel" in code[:200].lower():
            # Update compute kernel
            kernels_dir = example_path / "kernels" / "compute"
            for cpp_file in kernels_dir.glob("*.cpp"):
                cpp_file.write_text(code)
                logger.info(f"Updated compute kernel: {cpp_file.name}")
                files_updated += 1
                break

        elif "reader" in lang_tag.lower() or "// reader kernel" in code[:200].lower():
            # Update reader kernel
            kernels_dir = example_path / "kernels" / "dataflow"
            for cpp_file in kernels_dir.glob("*reader*.cpp"):
                cpp_file.write_text(code)
                logger.info(f"Updated reader kernel: {cpp_file.name}")
                files_updated += 1
                break

        elif "writer" in lang_tag.lower() or "// writer kernel" in code[:200].lower():
            # Update writer kernel
            kernels_dir = example_path / "kernels" / "dataflow"
            for cpp_file in kernels_dir.glob("*writer*.cpp"):
                cpp_file.write_text(code)
                logger.info(f"Updated writer kernel: {cpp_file.name}")
                files_updated += 1
                break

        elif "cpp" in lang_tag or "#include" in code[:200]:
            # Likely host code
            for cpp_file in example_path.glob("*.cpp"):
                cpp_file.write_text(code)
                logger.info(f"Updated host code: {cpp_file.name}")
                files_updated += 1
                break
        else:
            logger.warning(f"Could not determine file type for block with lang_tag='{lang_tag}'")
            logger.warning(f"Code preview: {code[:200]}")

    logger.info(f"Total files updated: {files_updated}")
