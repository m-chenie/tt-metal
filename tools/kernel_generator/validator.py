#!/usr/bin/env python3
"""
Validator for generated TT-Metal kernels
Tests compilation and basic functionality of generated kernels.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from config import TT_METAL_HOME, COMPILE_TIMEOUT, BUILD_TYPE

logger = logging.getLogger(__name__)


class KernelValidator:
    """Validates generated TT-Metal kernels"""

    def __init__(self):
        self.build_dir = TT_METAL_HOME / f"build_{BUILD_TYPE}"

    def validate_project(self, project_dir: Path, operation: str, core_mode: str) -> Dict[str, any]:
        """Validate a complete generated project"""

        logger.info(f"Validating {operation} {core_mode}-core project at {project_dir}")

        results = {
            "compilation": {"success": False, "errors": []},
            "kernel_syntax": {"success": True, "errors": []},
            "file_structure": {"success": True, "missing_files": []},
            "execution": {"success": False, "output": ""},
        }

        # Check file structure
        results["file_structure"] = self._validate_file_structure(project_dir, core_mode)

        # Check kernel syntax
        results["kernel_syntax"] = self._validate_kernel_syntax(project_dir)

        # Attempt compilation
        results["compilation"] = self._validate_compilation(project_dir)

        # If compilation succeeds, try execution
        if results["compilation"]["success"]:
            results["execution"] = self._validate_execution(project_dir)

        # Overall success
        results["overall_success"] = (
            results["file_structure"]["success"]
            and results["kernel_syntax"]["success"]
            and results["compilation"]["success"]
        )

        return results

    def _validate_file_structure(self, project_dir: Path, core_mode: str) -> Dict[str, any]:
        """Check if all required files are present"""

        required_files = [
            "CMakeLists.txt",
        ]

        # Add expected kernel files based on core mode
        if core_mode == "single":
            required_files.extend(
                [
                    "kernels/compute/*.cpp",
                    "kernels/dataflow/reader_binary_1_tile.cpp",
                    "kernels/dataflow/writer_1_tile.cpp",
                ]
            )
        else:
            required_files.extend(
                [
                    "kernels/compute/*.cpp",
                    "kernels/dataflow/reader_binary_tiles_partitioned.cpp",
                    "kernels/dataflow/writer_tiles_partitioned.cpp",
                ]
            )

        # Check for host code
        host_files = list(project_dir.glob("*.cpp"))
        if host_files:
            required_files.append("*.cpp")

        missing_files = []
        for pattern in required_files:
            if "*" in pattern:
                # Handle glob patterns
                matches = list(project_dir.glob(pattern))
                if not matches:
                    missing_files.append(pattern)
            else:
                # Handle exact file paths
                file_path = project_dir / pattern
                if not file_path.exists():
                    missing_files.append(pattern)

        return {
            "success": len(missing_files) == 0,
            "missing_files": missing_files,
        }

    def _validate_kernel_syntax(self, project_dir: Path) -> Dict[str, any]:
        """Basic syntax validation for kernel files"""

        errors = []
        kernel_files = []

        # Find all kernel files
        kernel_files.extend(project_dir.glob("kernels/**/*.cpp"))

        for kernel_file in kernel_files:
            try:
                content = kernel_file.read_text()

                # Basic syntax checks
                if not content.strip():
                    errors.append(f"{kernel_file.name}: Empty file")
                    continue

                # Check for required includes
                if "#include" not in content:
                    errors.append(f"{kernel_file.name}: No #include statements found")

                # Check for main function structure
                if "void MAIN" not in content and "namespace NAMESPACE" not in content:
                    errors.append(f"{kernel_file.name}: Missing MAIN function or NAMESPACE")

                # Check for circular buffer usage
                if "kernels/compute" in str(kernel_file):
                    if "cb_wait_front" not in content and "cb_reserve_back" not in content:
                        errors.append(f"{kernel_file.name}: No circular buffer operations found")

                # Check for common API functions
                if "kernels/dataflow" in str(kernel_file):
                    if "noc_async" not in content:
                        errors.append(f"{kernel_file.name}: No NOC async operations found")

            except Exception as e:
                errors.append(f"{kernel_file.name}: Error reading file - {str(e)}")

        return {
            "success": len(errors) == 0,
            "errors": errors,
        }

    def _validate_compilation(self, project_dir: Path) -> Dict[str, any]:
        """Attempt to compile the project"""

        try:
            # Create build directory
            build_dir = project_dir / "build"
            build_dir.mkdir(exist_ok=True)

            # Configure with CMake
            configure_cmd = [
                "cmake",
                "-S",
                str(project_dir),
                "-B",
                str(build_dir),
                f"-DTT_METAL_HOME={TT_METAL_HOME}",
                f"-DCMAKE_BUILD_TYPE={BUILD_TYPE}",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            ]

            configure_result = subprocess.run(
                configure_cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT, cwd=TT_METAL_HOME
            )

            if configure_result.returncode != 0:
                return {
                    "success": False,
                    "errors": [configure_result.stderr],
                    "stage": "configure",
                    "output": configure_result.stdout + configure_result.stderr,
                }

            # Build
            build_cmd = ["cmake", "--build", str(build_dir), "--parallel", "4"]

            build_result = subprocess.run(
                build_cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT, cwd=TT_METAL_HOME
            )

            return {
                "success": build_result.returncode == 0,
                "errors": [] if build_result.returncode == 0 else [build_result.stderr],
                "stage": "build",
                "output": build_result.stdout + build_result.stderr,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "errors": ["Compilation timeout"],
                "stage": "timeout",
                "output": "Process timed out",
            }
        except Exception as e:
            return {"success": False, "errors": [str(e)], "stage": "error", "output": f"Exception: {str(e)}"}

    def _validate_execution(self, project_dir: Path) -> Dict[str, any]:
        """Attempt to run the compiled executable"""

        try:
            # Find the executable
            build_dir = project_dir / "build"
            executables = list(build_dir.glob("*"))
            executables = [f for f in executables if f.is_file() and os.access(f, os.X_OK)]

            if not executables:
                return {
                    "success": False,
                    "output": "No executable found in build directory",
                    "errors": ["Missing executable"],
                }

            # Try to run the first executable
            exe_path = executables[0]

            run_result = subprocess.run(
                [str(exe_path)],
                capture_output=True,
                text=True,
                timeout=30,  # Short timeout for basic execution test
                cwd=project_dir,
            )

            return {
                "success": run_result.returncode == 0,
                "output": run_result.stdout + run_result.stderr,
                "errors": [] if run_result.returncode == 0 else [f"Exit code: {run_result.returncode}"],
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "output": "Execution timeout", "errors": ["Timeout during execution"]}
        except Exception as e:
            return {"success": False, "output": f"Exception: {str(e)}", "errors": [str(e)]}


def validate_generated_project(project_dir: Path, operation: str, core_mode: str) -> Dict[str, any]:
    """Convenience function to validate a generated project"""
    validator = KernelValidator()
    return validator.validate_project(project_dir, operation, core_mode)
