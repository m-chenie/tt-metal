"""
Iterative refinement: compile generated code, feed errors back to LLM, regenerate.
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from config import TT_METAL_HOME, MODEL_DEFAULT, TEMPERATURE, MAX_TOKENS
from host_api_retriever import retrieve_host_api_signatures, create_host_template
from retriever import retrieve_host_examples

logger = logging.getLogger("kernel_generator_v2")


def run_example(example_path: Path) -> Dict[str, any]:
    """
    Run the compiled example executable.

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
    # Read CMakeLists.txt to get the actual executable name
    cmake_file = example_path / "CMakeLists.txt"
    exec_name = None

    if cmake_file.exists():
        cmake_content = cmake_file.read_text()
        # Look for add_executable(name) or project(name)
        import re

        # Try add_executable first
        match = re.search(r"add_executable\s*\(\s*(\w+)", cmake_content)
        if match:
            exec_name = match.group(1)
        else:
            # Fall back to project name
            match = re.search(r"project\s*\(\s*(\w+)", cmake_content)
            if match:
                exec_name = match.group(1)

    if not exec_name:
        # Fallback: try to infer from directory name
        example_name = example_path.name
        if example_name.endswith("_generated"):
            base_name = example_name.replace("_generated", "")
        else:
            base_name = example_name
        exec_name = base_name.replace("_single_core", "_single").replace("_multi_core", "_multi")

    executable = TT_METAL_HOME / "build" / "programming_examples" / exec_name

    logger.info(f"Looking for executable: {executable}")

    if not executable.exists():
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Executable not found: {executable}",
            "exit_code": -1,
            "error_summary": f"Executable not found: {executable}",
        }

    logger.info(f"Running: {executable}")

    try:
        result = subprocess.run(
            [str(executable)],
            cwd=TT_METAL_HOME,
            capture_output=True,
            text=True,
            timeout=60,  # 1 minute timeout for runtime
        )

        # Combine stdout and stderr
        combined_output = result.stdout + "\n" + result.stderr

        # Check for runtime errors
        success = result.returncode == 0

        if not success:
            error_summary = parse_runtime_errors(combined_output)
        else:
            error_summary = ""

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
            "stderr": "Runtime timeout (>1 minute)",
            "combined": "Runtime timeout (>1 minute)",
            "exit_code": -1,
            "error_summary": "Program execution took too long - possible infinite loop or hang",
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "combined": str(e),
            "exit_code": -1,
            "error_summary": f"Failed to run executable: {e}",
        }


def parse_runtime_errors(output: str) -> str:
    """
    Parse runtime errors and extract detailed compilation error messages.

    Extracts:
    - Detailed compilation errors (missing functions, syntax errors, etc.)
    - TT_FATAL/TT_THROW errors with context
    - Assertion failures with details
    - Segmentation faults
    - Device errors

    Args:
        output: Combined stdout + stderr from runtime

    Returns:
        Detailed error summary with actionable information
    """
    if not output or not output.strip():
        return "No error output captured"

    lines = output.split("\n")

    # First, extract detailed compilation errors from kernel builds
    compilation_errors = []
    in_error_block = False
    current_error = []

    for i, line in enumerate(lines):
        # Detect start of compilation error blocks
        if "error:" in line.lower() and (".cpp" in line or ".h" in line):
            in_error_block = True
            current_error = [line.strip()]
        elif in_error_block:
            # Continue collecting error context
            if line.strip() and not line.strip().startswith("2025-"):  # Not a log timestamp line
                current_error.append(line.strip())
                # Stop if we see another error or a new section
                if len(current_error) > 10 or "error:" in line.lower() and len(current_error) > 1:
                    compilation_errors.append("\n".join(current_error))
                    current_error = [line.strip()] if "error:" in line.lower() else []
            else:
                if current_error:
                    compilation_errors.append("\n".join(current_error))
                    current_error = []
                in_error_block = False

    if current_error:
        compilation_errors.append("\n".join(current_error))

    # Extract other error types
    errors = {"tt_fatal": [], "tt_throw": [], "assertions": [], "device": [], "other": []}

    for i, line in enumerate(lines):
        lower_line = line.lower()

        # TT_FATAL errors - get more context
        if "tt_fatal" in lower_line:
            context_lines = [line.strip()]
            for j in range(i + 1, min(i + 3, len(lines))):
                if lines[j].strip():
                    context_lines.append(lines[j].strip())
            errors["tt_fatal"].append("\n".join(context_lines))

        # TT_THROW errors - get the full error message
        elif "tt_throw" in lower_line:
            context_lines = [line.strip()]
            # Get next few lines for full error context
            for j in range(i + 1, min(i + 5, len(lines))):
                if lines[j].strip() and not lines[j].strip().startswith("2025-"):
                    context_lines.append(lines[j].strip())
                elif "Log:" in lines[j]:
                    # Get the log details
                    for k in range(j, min(j + 10, len(lines))):
                        if lines[k].strip():
                            context_lines.append(lines[k].strip())
                        if "error:" in lines[k].lower():
                            break
                    break
            errors["tt_throw"].append("\n".join(context_lines))

        # Assertion failures - get full details
        elif "assertion" in lower_line and "failed" in lower_line:
            context_lines = [line.strip()]
            for j in range(i + 1, min(i + 3, len(lines))):
                if "note:" in lines[j].lower() or "comparison reduces" in lines[j].lower():
                    context_lines.append(lines[j].strip())
            errors["assertions"].append("\n".join(context_lines))

    # Build detailed summary
    summary_parts = []

    if compilation_errors:
        summary_parts.append("=" * 80)
        summary_parts.append("DETAILED COMPILATION ERRORS:")
        summary_parts.append("=" * 80)
        for err in compilation_errors[:20]:  # Show up to 20 distinct errors
            summary_parts.append(err)
            summary_parts.append("-" * 40)

    if errors["assertions"]:
        summary_parts.append("\nASSERTION FAILURES:")
        for err in set(errors["assertions"][:10]):  # Unique assertions
            summary_parts.append(err)
            summary_parts.append("")

    if errors["tt_throw"]:
        summary_parts.append(f"\nTT_THROW ERRORS:")
        for err in errors["tt_throw"][:5]:  # First 5 with context
            summary_parts.append(err)
            summary_parts.append("")

    if errors["tt_fatal"]:
        summary_parts.append(f"\nFATAL ERRORS:")
        for err in errors["tt_fatal"][:5]:
            summary_parts.append(err)
            summary_parts.append("")

    if not compilation_errors and not any(errors.values()):
        # No categorized errors, show last 50 lines of output
        summary_parts.append("RUNTIME FAILURE - Last 50 lines of output:")
        summary_parts.extend(lines[-50:])

    return "\n".join(summary_parts)


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
    Parse and extract detailed compilation errors with full context.

    Extracts:
    - Complete error messages with file:line information
    - Compiler suggestions (did you mean X?)
    - Linker errors (undefined reference)
    - Missing include errors
    - API signature mismatches
    - Template instantiation errors

    Args:
        output: Combined stdout + stderr from build

    Returns:
        Detailed error summary with actionable information
    """
    if not output or not output.strip():
        return "No error output captured"

    lines = output.split("\n")

    # Extract detailed compilation errors with context
    detailed_errors = []
    in_error_block = False
    current_error = []

    for i, line in enumerate(lines):
        lower_line = line.lower()

        # Start of an error
        if "error:" in lower_line and (".cpp" in line or ".h" in line or "instantiation of" in lower_line):
            if current_error:
                detailed_errors.append("\n".join(current_error))
            current_error = [line.strip()]
            in_error_block = True
        elif in_error_block:
            # Collect context lines (notes, suggestions, code snippets)
            if any(
                keyword in lower_line
                for keyword in ["note:", "required", "did you mean", "^~~", "instantiation", "comparison reduces"]
            ):
                current_error.append(line.strip())
            elif line.strip().startswith("|") or "^" in line:
                # Code context or error markers
                current_error.append(line.strip())
            elif "error:" in lower_line:
                # New error starting, save current and start new
                detailed_errors.append("\n".join(current_error))
                current_error = [line.strip()]
            elif not line.strip():
                # Empty line might signal end of error block
                if len(current_error) > 1:
                    detailed_errors.append("\n".join(current_error))
                    current_error = []
                in_error_block = False
            elif len(current_error) < 15:  # Don't let errors get too long
                current_error.append(line.strip())

    if current_error:
        detailed_errors.append("\n".join(current_error))

    # Also categorize for summary
    errors = {"linker": [], "includes": [], "api": [], "cmake": []}

    for line in lines:
        lower_line = line.lower()

        if "undefined reference" in lower_line:
            match = re.search(r"undefined reference to [`']([^'`]+)", line)
            if match:
                symbol = match.group(1)
                simple_symbol = symbol.split("(")[0].split("<")[0]
                if simple_symbol not in errors["linker"]:
                    errors["linker"].append(simple_symbol)

        elif "fatal error:" in lower_line and (".h" in lower_line or ".hpp" in lower_line):
            match = re.search(r"fatal error: ([^:]+): No such file", line)
            if match:
                errors["includes"].append(match.group(1))

        elif "cmake error" in lower_line:
            if line.strip():
                errors["cmake"].append(line.strip()[:200])

    # Build detailed summary
    summary_parts = []

    if detailed_errors:
        summary_parts.append("=" * 80)
        summary_parts.append("DETAILED COMPILATION ERRORS:")
        summary_parts.append("=" * 80)
        for i, err in enumerate(detailed_errors[:25], 1):  # Show up to 25 errors
            summary_parts.append(f"\nError #{i}:")
            summary_parts.append(err)
            summary_parts.append("-" * 40)

    if errors["linker"]:
        summary_parts.append(f"\n\nLINKER ERRORS ({len(errors['linker'])} unique symbols):")
        for symbol in errors["linker"][:10]:
            summary_parts.append(f"  - undefined reference to: {symbol}")

    if errors["includes"]:
        summary_parts.append(f"\n\nINCLUDE ERRORS:")
        for inc in set(errors["includes"]):
            summary_parts.append(f"  - missing header: {inc}")

    if errors["cmake"]:
        summary_parts.append(f"\n\nCMAKE ERRORS:")
        for err in errors["cmake"][:5]:
            summary_parts.append(f"  - {err}")

    if not detailed_errors and not any(errors.values()):
        # No categorized errors, show last 30 lines of output
        summary_parts.append("BUILD FAILED - Last 30 lines of error output:")
        summary_parts.extend(lines[-30:])

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
    current_files: Dict[str, str],
    error_summary: str,
    error_type: str,
    iteration: int,
    example_path: Path = None,
    previous_prompt: str = None,
) -> Tuple[str, str]:
    """
    Build system and user prompts for fixing compilation errors.

    Args:
        current_files: Dict with keys like 'compute', 'reader', 'writer', 'host_code', 'cmake'
        error_summary: Parsed error summary
        error_type: Primary error type (linker/syntax/etc)
        iteration: Current iteration number
        example_path: Path to example (used to load original prompt)
        previous_prompt: The prompt from the previous iteration (if iteration > 1)

    Returns:
        (system_prompt, user_prompt)
    """
    # For iteration 1, load the original generation prompt
    # For iteration N > 1, use the previous refinement prompt
    context_prompt = ""
    if iteration == 1 and example_path:
        original_prompt_file = example_path / "original_generation_prompt.md"
        if original_prompt_file.exists():
            context_prompt = original_prompt_file.read_text()
            logger.info("Loaded original generation prompt for iteration 1")
    elif previous_prompt:
        context_prompt = previous_prompt
        logger.info(f"Using previous iteration prompt for iteration {iteration}")

    system_prompt = f"""You are an expert TT-Metal developer fixing compilation errors.

This is iteration {iteration} of debugging.

{'='*80}
PREVIOUS REQUEST (what you were asked to do):
{'='*80}
{context_prompt if context_prompt else "Not available - this is the first iteration"}

{'='*80}
YOUR PREVIOUS OUTPUT (the code you generated):
{'='*80}
"""

    # Show ALL the code that was generated previously
    for key in ["compute", "reader", "writer"]:
        if key in current_files:
            system_prompt += f"\n## {key.title()} Kernel\n"
            system_prompt += f"```cpp\n{current_files[key]}\n```\n"

    if "host_code" in current_files:
        system_prompt += "\n## Host Code\n"
        system_prompt += f"```cpp\n{current_files['host_code']}\n```\n"

    system_prompt += "\n## CMakeLists.txt\n"
    system_prompt += f"```cmake\n{current_files.get('cmake', 'N/A')}\n```\n"

    system_prompt += f"""
{'='*80}
COMPILATION ERRORS (what went wrong):
{'='*80}
```
{error_summary}
```
"""

    # Simple, direct user prompt
    user_prompt = f"""The code you generated in the previous iteration failed to compile with the errors shown above.

Please fix ALL the errors and provide the complete, corrected code.

Output format - provide ALL files with fixes applied:

```cpp
// COMPUTE KERNEL
[complete corrected compute kernel code]
```

```cpp
// READER KERNEL
[complete corrected reader kernel code]
```

```cpp
// WRITER KERNEL
[complete corrected writer kernel code]
```

```cpp
// HOST CODE
[complete corrected host code]
```

```cmake
# CMakeLists.txt
[complete corrected cmake file]
```

CRITICAL: Provide COMPLETE files, not just the changed sections."""

    return system_prompt, user_prompt


def iterative_refine(client: Any, args):
    """
    Main iterative refinement loop.

    Args:
        client: LLM API client (Groq or OpenAI)
        args: Parsed command-line arguments
    """
    # Pre-flight check: verify API key is set
    if not hasattr(client, "api_key") or not client.api_key or client.api_key == "":
        raise SystemExit(
            "ERROR: LLM API key not set. Please set GROQ_API_KEY or OPENAI_API_KEY environment variable.\n"
            "Example: export OPENAI_API_KEY='your-api-key-here'"
        )

    example_path = Path(args.example_path)

    if not example_path.exists():
        raise SystemExit(f"Example path does not exist: {example_path}")

    logger.info(f"Starting iterative refinement on: {example_path}")
    logger.info(f"Max iterations: {args.max_iterations}")

    # Track the previous iteration's prompt for context
    previous_iteration_prompt = None

    # Initial compilation attempt
    for iteration in range(1, args.max_iterations + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}/{args.max_iterations}")
        logger.info(f"{'='*60}\n")

        # Compile
        compile_result = compile_example(example_path)

        if compile_result["success"]:
            logger.info("✓ Compilation successful!")

            # Now try to run the executable
            logger.info("Attempting to run the executable...")
            run_result = run_example(example_path)

            if run_result["success"]:
                logger.info("✓ ✓ ✓ RUNTIME SUCCESSFUL! ✓ ✓ ✓")
                logger.info(f"Example compiled and ran successfully after {iteration} iteration(s)")
                return
            else:
                logger.warning(f"✗ Runtime failed (exit code: {run_result['exit_code']})")

                # Save runtime output to file
                runtime_file = example_path / f"runtime_output_iteration_{iteration}.txt"
                runtime_file.write_text(
                    f"=== STDOUT ===\n{run_result['stdout']}\n\n=== STDERR ===\n{run_result['stderr']}\n\n=== COMBINED ===\n{run_result.get('combined', '')}"
                )
                logger.info(f"Saved runtime output to: {runtime_file}")

                logger.info("\nRuntime Error Summary:")
                logger.info(run_result["error_summary"])

                # Use runtime errors for refinement
                error_type = "runtime"
                error_summary = run_result["error_summary"]
        else:
            logger.warning(f"✗ Compilation failed (exit code: {compile_result['exit_code']})")

            # Debug: Save full output to file
            debug_file = example_path / f"build_output_iteration_{iteration}.txt"
            debug_file.write_text(
                f"=== STDOUT ===\n{compile_result['stdout']}\n\n=== STDERR ===\n{compile_result['stderr']}\n\n=== COMBINED ===\n{compile_result.get('combined', '')}"
            )
            logger.info(f"Saved full build output to: {debug_file}")

            logger.info("\nCompilation Error Summary:")
            logger.info(compile_result["error_summary"])

            # Detect error type
            error_type = detect_error_type(compile_result["error_summary"])
            error_summary = compile_result["error_summary"]

        logger.info(f"\nDetected error type: {error_type}")

        # Load current files
        current_files = load_current_files(example_path)

        # Build refinement prompt
        system_prompt, user_prompt = build_refinement_prompt(
            current_files, error_summary, error_type, iteration, example_path, previous_iteration_prompt
        )

        # Save the current prompt for use in the next iteration
        current_iteration_prompt = f"## System\n```\n{system_prompt}\n```\n\n## User\n```\n{user_prompt}\n```\n"

        if args.save_prompt:
            prompt_file = example_path / f"refinement_iteration_{iteration}.md"
            prompt_file.write_text(f"# Iteration {iteration}\n\n{current_iteration_prompt}")
            logger.info(f"Saved refinement prompt to: {prompt_file}")

        # Call LLM to fix errors
        logger.info("Calling LLM for fixes...")
        try:
            # Use the model from args if provided, otherwise use default
            # For OpenAI provider, default to gpt-4o if not specified
            model_to_use = args.model
            if not model_to_use or model_to_use == MODEL_DEFAULT:
                if args.provider == "openai":
                    model_to_use = "gpt-4o-2024-08-06"
                else:
                    model_to_use = MODEL_DEFAULT

            resp = client.chat.completions.create(
                model=model_to_use,
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

            # Save this iteration's full prompt for the next iteration
            previous_iteration_prompt = current_iteration_prompt

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
