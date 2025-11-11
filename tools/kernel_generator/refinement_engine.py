#!/usr/bin/env python3
"""
Iterative Refinement Engine for TT-Metal Kernel Generator
Compiles generated kernels and iteratively fixes compilation errors using LLM feedback.
"""

import subprocess
import re
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from openai import OpenAI
from config import (
    TT_METAL_HOME,
    COMPILE_TIMEOUT,
    MAX_REFINEMENT_ITERATIONS,
    BUILD_TYPE,
    OPENAI_TEMPERATURE,
    OPENAI_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


class RefinementEngine:
    """Handles iterative refinement of generated kernels based on compilation feedback"""

    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.build_dir = TT_METAL_HOME / f"build_{BUILD_TYPE}"

    def refine_kernels(
        self,
        kernels: Dict[str, str],
        host_code: str,
        cmake_content: str,
        operation: str,
        core_mode: str,
        system_prompt: str,
        output_dir: Path,
        max_iterations: Optional[int] = None,
    ) -> Tuple[Dict[str, str], str, str, List[str]]:
        """
        Iteratively refine kernels by compiling and fixing errors

        Returns:
            - Refined kernels dict
            - Refined host code
            - Refined cmake content
            - List of refinement logs
        """
        max_iters = max_iterations if max_iterations is not None else MAX_REFINEMENT_ITERATIONS
        logger.info(
            f"Starting iterative refinement for {operation} {core_mode}-core kernels (max {max_iters} iterations)"
        )

        refinement_logs = []
        current_kernels = kernels.copy()
        current_host = host_code
        current_cmake = cmake_content

        for iteration in range(max_iters):
            logger.info(f"Refinement iteration {iteration + 1}/{max_iters}")

            # Create temporary project
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                project_path = self._create_temp_project(
                    temp_path, current_kernels, current_host, current_cmake, operation, core_mode
                )

                # Attempt compilation
                compile_result = self._compile_project(project_path)

                if compile_result["success"]:
                    logger.info(f"Compilation successful after {iteration + 1} iterations")
                    refinement_logs.append(f"Iteration {iteration + 1}: Compilation successful")
                    break

                # Extract and analyze errors
                errors = self._extract_compilation_errors(compile_result["output"])
                if not errors:
                    logger.warning("Compilation failed but no specific errors found")
                    break

                logger.info(f"Found {len(errors)} compilation errors")
                refinement_logs.append(f"Iteration {iteration + 1}: {len(errors)} errors found")

                # Use LLM to fix errors
                fixes = self._generate_fixes(
                    errors, current_kernels, current_host, current_cmake, system_prompt, operation, core_mode
                )

                if not fixes:
                    logger.warning("LLM could not generate fixes")
                    break

                # Apply fixes
                current_kernels, current_host, current_cmake = self._apply_fixes(
                    fixes, current_kernels, current_host, current_cmake
                )

                refinement_logs.append(f"Iteration {iteration + 1}: Applied {len(fixes)} fixes")

        return current_kernels, current_host, current_cmake, refinement_logs

    def _create_temp_project(
        self,
        temp_path: Path,
        kernels: Dict[str, str],
        host_code: str,
        cmake_content: str,
        operation: str,
        core_mode: str,
    ) -> Path:
        """Create a temporary project for compilation testing"""
        project_name = f"{operation}_{core_mode}_test"
        project_path = temp_path / project_name

        # Create directory structure
        (project_path / "kernels" / "compute").mkdir(parents=True)
        (project_path / "kernels" / "dataflow").mkdir(parents=True)

        # Write kernel files
        for kernel_type, code in kernels.items():
            if kernel_type == "compute":
                kernel_file = project_path / "kernels" / "compute" / f"{operation}_tiles.cpp"
            elif kernel_type == "reader":
                if core_mode == "single":
                    kernel_file = project_path / "kernels" / "dataflow" / "reader_binary_1_tile.cpp"
                else:
                    kernel_file = project_path / "kernels" / "dataflow" / "reader_binary_tiles_partitioned.cpp"
            elif kernel_type == "writer":
                if core_mode == "single":
                    kernel_file = project_path / "kernels" / "dataflow" / "writer_1_tile.cpp"
                else:
                    kernel_file = project_path / "kernels" / "dataflow" / "writer_tiles_partitioned.cpp"

            kernel_file.write_text(code)

        # Write host code
        host_file = project_path / f"{project_name}.cpp"
        host_file.write_text(host_code)

        # Write CMakeLists.txt
        cmake_file = project_path / "CMakeLists.txt"
        cmake_file.write_text(cmake_content)

        return project_path

    def _compile_project(self, project_path: Path) -> Dict[str, any]:
        """Attempt to compile the project using TT-Metal build system"""
        try:
            # Use the TT-Metal build system
            logger.info(f"Compiling project at {project_path}")

            # Try to build the specific programming example
            # First, try using the TT-Metal build script
            build_cmd = [str(TT_METAL_HOME / "build_metal.sh"), "--build-programming-examples"]

            build_result = subprocess.run(
                build_cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT, cwd=TT_METAL_HOME
            )

            # Check the output for our specific example
            output_text = build_result.stderr + build_result.stdout

            # Look for errors related to our specific example
            example_name = project_path.name
            if example_name in output_text and ("error:" in output_text.lower() or "undefined" in output_text.lower()):
                success = False
            else:
                # If the build completed without errors mentioning our example, consider it successful
                success = build_result.returncode == 0

            return {
                "success": success,
                "output": output_text,
                "stage": "build",
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "output": "Compilation timed out", "stage": "timeout"}
        except Exception as e:
            return {"success": False, "output": f"Compilation error: {str(e)}", "stage": "error"}

    def _extract_compilation_errors(self, compile_output: str) -> List[Dict[str, str]]:
        """Extract structured compilation errors from build output"""
        errors = []

        # Common error patterns
        error_patterns = [
            r"(.+):(\d+):(\d+):\s*error:\s*(.+)",  # GCC/Clang error format
            r"(.+)\((\d+)\):\s*error\s*C\d+:\s*(.+)",  # MSVC error format
            r"CMake Error at (.+):(\d+) \((.+)\):\s*(.+)",  # CMake errors
            r"CMake Error:\s*(.+)",  # CMake generic errors
            r"Error:\s*(.+)",  # Generic error
            r"undefined reference to\s*[`'](.+)'",  # Linker errors
            r"(.+):\s*undefined symbol:\s*(.+)",  # Symbol errors
            r"include could not find requested file:\s*(.+)",  # CMake include errors
            r"Unknown CMake command\s*[\"'](.+)[\"']",  # CMake command errors
        ]

        lines = compile_output.split("\n")

        for line in lines:
            for pattern in error_patterns:
                match = re.search(pattern, line)
                if match:
                    if len(match.groups()) >= 4:  # File, line, column, message
                        errors.append(
                            {
                                "file": match.group(1),
                                "line": match.group(2),
                                "column": match.group(3),
                                "message": match.group(4),
                                "full_line": line.strip(),
                            }
                        )
                    elif len(match.groups()) >= 2:  # Simpler format
                        errors.append(
                            {
                                "file": "unknown",
                                "line": "unknown",
                                "column": "unknown",
                                "message": match.group(1) if len(match.groups()) == 1 else match.group(2),
                                "full_line": line.strip(),
                            }
                        )
                    break

        return errors[:10]  # Limit to first 10 errors to avoid overwhelming LLM

    def _generate_fixes(
        self,
        errors: List[Dict[str, str]],
        kernels: Dict[str, str],
        host_code: str,
        cmake_content: str,
        system_prompt: str,
        operation: str,
        core_mode: str,
    ) -> Optional[Dict[str, str]]:
        """Use LLM to generate fixes for compilation errors"""

        # Format errors for LLM
        error_summary = "\\n".join([f"- {error['file']}:{error['line']} - {error['message']}" for error in errors])

        fix_prompt = f"""The generated {operation} {core_mode}-core kernels have compilation errors. Please provide fixes.

## Compilation Errors:
{error_summary}

## Current Code:

### Compute Kernel:
```cpp
{kernels.get('compute', 'Not generated')}
```

### Reader Kernel:
```cpp
{kernels.get('reader', 'Not generated')}
```

### Writer Kernel:
```cpp
{kernels.get('writer', 'Not generated')}
```

### Host Code:
```cpp
{host_code}
```

## Instructions:
1. Analyze each compilation error carefully
2. Identify the root cause (missing headers, API misuse, syntax errors, etc.)
3. Provide corrected versions of the affected files
4. Ensure the fixes maintain the original functionality
5. Use only valid TT-Metal APIs and patterns from the knowledge base

Return your response as JSON with this structure:
{{
    "fixes": [
        {{
            "file_type": "compute|reader|writer|host|cmake",
            "error_analysis": "Brief explanation of the error",
            "fixed_code": "Complete corrected code for the file"
        }}
    ]
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": fix_prompt}],
                temperature=OPENAI_TEMPERATURE,
                max_tokens=OPENAI_MAX_TOKENS,
            )

            # Parse JSON response
            import json

            response_text = response.choices[0].message.content

            # Extract JSON from response (handle markdown formatting)
            if "```json" in response_text:
                json_content = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_content = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_content = response_text

            fixes_data = json.loads(json_content)
            return fixes_data.get("fixes", [])

        except Exception as e:
            logger.error(f"Error generating fixes: {e}")
            return None

    def _apply_fixes(
        self, fixes: List[Dict[str, str]], kernels: Dict[str, str], host_code: str, cmake_content: str
    ) -> Tuple[Dict[str, str], str, str]:
        """Apply LLM-generated fixes to the code"""

        updated_kernels = kernels.copy()
        updated_host = host_code
        updated_cmake = cmake_content

        for fix in fixes:
            file_type = fix.get("file_type", "")
            fixed_code = fix.get("fixed_code", "")

            if not fixed_code:
                continue

            if file_type in ["compute", "reader", "writer"]:
                updated_kernels[file_type] = fixed_code
                logger.info(f"Applied fix to {file_type} kernel")
            elif file_type == "host":
                updated_host = fixed_code
                logger.info("Applied fix to host code")
            elif file_type == "cmake":
                updated_cmake = fixed_code
                logger.info("Applied fix to CMakeLists.txt")

        return updated_kernels, updated_host, updated_cmake
