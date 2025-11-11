#!/usr/bin/env python3
"""
TT-Metal Kernel Generator (Clean Version)
Generates only kernels using RAG. User handles host code and CMakeLists.txt manually.
If compilation fails, use --refine flag with path to fix issues.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple
import argparse
import re
from pathlib import Path
from datetime import datetime
from groq import Groq
from rag_knowledge_base import RAGKnowledgeBase
from host_code_generator import HostCodeGenerator
from config import (
    OPENAI_MODEL_DEFAULT,
    OPENAI_TEMPERATURE,
    OPENAI_MAX_TOKENS,
    TT_METAL_HOME,
    OPERATIONS,
    CORE_MODES,
    MAX_REFINEMENT_ITERATIONS,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Groq client
groq_api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
client = Groq(api_key=groq_api_key)


class KernelGenerator:
    """Main kernel generator with iterative refinement"""

    def __init__(self, groq_api_key: str, model: str = OPENAI_MODEL_DEFAULT):
        self.client = Groq(api_key=groq_api_key)
        self.model = model
        self.rag_kb = RAGKnowledgeBase()
        self.host_generator = HostCodeGenerator(self.client)
        # Initialize RefinementEngine
        from refinement_engine import RefinementEngine

        self.refinement_engine = RefinementEngine(self.client)

    def refine_existing_example(
        self,
        target_path: str,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, any]:
        """Refine an existing programming example by fixing compilation errors"""

        logger.info(f"Starting refinement of existing example: {target_path}")

        # Resolve full path
        if not os.path.isabs(target_path):
            # If relative path, assume it's under programming_examples
            full_path = TT_METAL_HOME / "tt_metal" / "programming_examples" / target_path
        else:
            full_path = Path(target_path)

        if not full_path.exists():
            logger.error(f"Target path does not exist: {full_path}")
            return {"success": False, "error": f"Path not found: {full_path}"}

        # Extract operation and core_mode from path or detect from files
        operation, core_mode = self._detect_operation_from_path(full_path)
        if not operation:
            logger.error("Could not detect operation type from path")
            return {"success": False, "error": "Could not detect operation type"}

        logger.info(f"Detected operation: {operation}, core_mode: {core_mode}")

        # Load existing files
        existing_files = self._load_existing_files(full_path)
        if not existing_files:
            logger.error("Could not load existing files")
            return {"success": False, "error": "Could not load existing files"}

        # Build RAG knowledge base for this operation
        logger.info("Building RAG knowledge base...")
        op_config = OPERATIONS.get(operation, {})
        operation_type = op_config.get("operation_type", "standard")
        knowledge_base = self.rag_kb.build_knowledge_base(core_mode, operation_type)
        system_prompt = self.rag_kb.get_system_prompt(knowledge_base, core_mode, operation)

        # Run iterative refinement
        logger.info("Starting iterative refinement...")
        refined_kernels, refined_host, refined_cmake, refinement_logs = self.refinement_engine.refine_kernels(
            existing_files["kernels"],
            existing_files["host_code"],
            existing_files["cmake_content"],
            operation,
            core_mode,
            system_prompt,
            full_path,
            max_iterations,
        )

        # Save refined files
        logger.info("Saving refined files...")
        self._save_generated_files(
            full_path, operation, core_mode, refined_kernels, refined_host, refined_cmake, refinement_logs
        )

        logger.info("Refinement complete!")

        return {
            "success": True,
            "output_dir": str(full_path),
            "kernels": refined_kernels,
            "host_code": refined_host,
            "refinement_logs": refinement_logs,
        }

    def _detect_operation_from_path(self, path: Path) -> Tuple[str, str]:
        """Detect operation and core_mode from path name"""
        path_name = path.name.lower()

        # Try to match against known operations
        for operation in OPERATIONS.keys():
            if operation in path_name:
                # Detect core mode
                if "multi" in path_name:
                    return operation, "multi"
                else:
                    return operation, "single"

        # If no direct match, try to detect from files
        return self._detect_operation_from_files(path)

    def _detect_operation_from_files(self, path: Path) -> Tuple[str, str]:
        """Detect operation from file contents"""
        # Look for compute kernel files
        compute_files = list(path.glob("**/compute/*.cpp"))
        if not compute_files:
            return "", ""

        # Read first compute file and look for operation patterns
        try:
            content = compute_files[0].read_text()
            # Look for specific operation patterns in the code
            for operation, config in OPERATIONS.items():
                if config.get("api_compute", "") in content:
                    # Detect core mode from content or file structure
                    if "multi" in path.name.lower() or "multicore" in content:
                        return operation, "multi"
                    else:
                        return operation, "single"
        except Exception as e:
            logger.warning(f"Error reading compute file: {e}")

        return "", ""

    def _load_existing_files(self, path: Path) -> Dict[str, any]:
        """Load existing kernel files, host code, and CMakeLists.txt"""
        files = {"kernels": {}, "host_code": "", "cmake_content": ""}

        try:
            # Load kernel files
            compute_files = list(path.glob("**/compute/*.cpp"))
            dataflow_files = list(path.glob("**/dataflow/*.cpp"))

            # Load compute kernel
            if compute_files:
                files["kernels"]["compute"] = compute_files[0].read_text()

            # Load dataflow kernels
            for df_file in dataflow_files:
                if "reader" in df_file.name.lower():
                    files["kernels"]["reader"] = df_file.read_text()
                elif "writer" in df_file.name.lower():
                    files["kernels"]["writer"] = df_file.read_text()

            # Load host code
            host_files = list(path.glob("*.cpp"))
            if host_files:
                files["host_code"] = host_files[0].read_text()

            # Load CMakeLists.txt
            cmake_files = list(path.glob("**/CMakeLists.txt"))
            if cmake_files:
                files["cmake_content"] = cmake_files[0].read_text()

            logger.info(
                f"Loaded {len(files['kernels'])} kernels, host code: {bool(files['host_code'])}, cmake: {bool(files['cmake_content'])}"
            )

        except Exception as e:
            logger.error(f"Error loading existing files: {e}")
            return {}

        return files

    def generate_kernels(
        self,
        operation: str,
        core_mode: str,
        output_dir: Optional[Path] = None,
        enable_refinement: bool = False,
        generate_host: bool = False,
        validate: bool = False,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, any]:
        """Main generation pipeline"""

        logger.info(f"Starting kernel generation for {operation} ({core_mode}-core)")

        # Set up output directory - put it in programming_examples
        if not output_dir:
            op_config = OPERATIONS[operation]
            core_config = CORE_MODES[core_mode]
            output_dir = (
                TT_METAL_HOME / "tt_metal" / "programming_examples" / f"{operation}{core_config['suffix']}_generated"
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        # Store output directory for debug file saving
        self._current_output_dir = output_dir

        # Generate initial kernels using smart RAG
        logger.info("Generating initial kernels...")
        kernels = self._generate_initial_kernels(operation, core_mode)

        if not kernels:
            logger.error("Failed to generate initial kernels")
            return {"success": False, "error": "Kernel generation failed"}

        # Generate host code if requested
        host_code = ""
        cmake_content = ""
        if generate_host:
            logger.info("Generating host code...")
            # Get system prompt for host code generation
            system_prompt = self.rag_kb.get_system_prompt_smart(f"host code {operation} {core_mode} core", core_mode)
            host_code = self.host_generator.generate_host_code(operation, core_mode, kernels, system_prompt, self.model)
            cmake_content = self.host_generator.generate_cmake_file(operation, core_mode)

        # Iterative refinement if enabled
        refinement_logs = []
        if enable_refinement and generate_host:
            logger.info("Starting iterative refinement...")
            # Reuse the system_prompt from host code generation
            kernels, host_code, cmake_content, refinement_logs = self.refinement_engine.refine_kernels(
                kernels, host_code, cmake_content, operation, core_mode, system_prompt, output_dir, max_iterations
            )

        # Save generated files
        logger.info("Saving generated files...")
        self._save_generated_files(output_dir, operation, core_mode, kernels, host_code, cmake_content, refinement_logs)

        # Validate if requested
        validation_results = {}
        if validate:
            logger.info("Running validation...")
            validation_results = self._validate_kernels(output_dir, operation, core_mode)

        logger.info("Kernel generation complete!")

        return {
            "success": True,
            "output_dir": str(output_dir),
            "kernels": kernels,
            "host_code": host_code,
            "refinement_logs": refinement_logs,
            "validation_results": validation_results,
        }

    def _generate_initial_kernels(self, operation: str, core_mode: str) -> Dict[str, str]:
        """Generate the initial set of kernels using smart RAG"""

        kernels = {}
        op_config = OPERATIONS[operation]

        # Generate compute kernel
        kernels["compute"] = self._generate_single_kernel("compute", operation, core_mode, op_config)

        # Generate reader kernel
        kernels["reader"] = self._generate_single_kernel("reader", operation, core_mode, op_config)

        # Generate writer kernel
        kernels["writer"] = self._generate_single_kernel("writer", operation, core_mode, op_config)

        # Check if all kernels were generated successfully
        if not all(kernels.values()):
            logger.error("Failed to generate one or more kernels")
            return {}

        return kernels

    def _generate_single_kernel(
        self, kernel_type: str, operation: str, core_mode: str, op_config: Dict[str, str]
    ) -> str:
        """Generate a single kernel using smart RAG"""

        operation_type = op_config.get("operation_type", "standard")

        # Create query for smart RAG
        query = f"{kernel_type} kernel {operation} {core_mode} core"

        # Get smart system prompt with relevant context only
        system_prompt = self.rag_kb.get_system_prompt_smart(query, core_mode)

        # Create specific prompt for this kernel type
        if kernel_type == "compute":
            if operation_type == "sfpu_chain":
                user_prompt = self._create_sfpu_chain_compute_prompt(operation, op_config)
            else:
                user_prompt = f"""Generate a TT-Metal compute kernel for {op_config['description']}.

Requirements:
- Use the {op_config['api_init']}() and {op_config['api_compute']}() functions
- Follow the {core_mode}-core patterns from the knowledge base
- Handle multiple tiles per core for multi-core mode
- Use circular buffers: CB_0 (input A), CB_1 (input B), CB_16 (output)
- Include proper error handling and comments
- Filename: kernels/compute/{op_config['kernel_name']}.cpp

Generate only the .cpp code."""

        elif kernel_type == "reader":
            if core_mode == "single":
                user_prompt = f"""Generate a TT-Metal reader kernel for single-core {op_config['description']}.

Requirements:
- Read input tiles from DRAM to CB_0 and CB_1
- Follow single-core reader patterns from knowledge base
- Use NOC async reads with barriers
- Runtime args: src0_addr, src1_addr
- Filename: kernels/dataflow/reader_binary_1_tile.cpp

Generate only the .cpp code."""
            else:
                user_prompt = f"""Generate a TT-Metal reader kernel for multi-core {op_config['description']}.

Requirements:
- Each core reads its assigned tiles from DRAM to CB_0 and CB_1
- Follow multi-core reader patterns from knowledge base
- Runtime args: src0_addr, src1_addr, Mt, Kt, Nt, start_tile_id, num_tiles
- Use NOC async reads: noc_async_read_tile, noc_async_read_barrier
- For element-wise: read same tile_id from both sources
- Filename: kernels/dataflow/reader_binary_tiles_partitioned.cpp

Generate only the .cpp code."""

        elif kernel_type == "writer":
            if core_mode == "single":
                user_prompt = f"""Generate a TT-Metal writer kernel for single-core {op_config['description']}.

Requirements:
- Write output tile from CB_16 to DRAM
- Follow single-core writer patterns from knowledge base
- Use NOC async write with barrier
- Runtime args: dst_addr
- Filename: kernels/dataflow/writer_1_tile.cpp

Generate only the .cpp code."""
            else:
                user_prompt = f"""Generate a TT-Metal writer kernel for multi-core {op_config['description']}.

Requirements:
- Each core writes its assigned result tiles from CB_16 to DRAM
- Follow multi-core writer patterns from knowledge base
- Runtime args: dst_addr, start_tile_id, num_tiles
- Use NOC async writes: noc_async_write_tile, noc_async_write_barrier
- Filename: kernels/dataflow/writer_tiles_partitioned.cpp

Generate only the .cpp code."""

        try:
            # Save full prompt to markdown file for debugging
            self._save_prompt_debug(kernel_type, operation, core_mode, system_prompt, user_prompt)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=OPENAI_TEMPERATURE,
                max_tokens=OPENAI_MAX_TOKENS,
            )

            generated_code = response.choices[0].message.content

            # Extract code from markdown if present
            if "```cpp" in generated_code:
                generated_code = generated_code.split("```cpp")[1].split("```")[0].strip()
            elif "```" in generated_code:
                generated_code = generated_code.split("```")[1].split("```")[0].strip()

            logger.info(f"Generated {kernel_type} kernel ({len(generated_code)} characters)")
            return generated_code

        except Exception as e:
            logger.error(f"Error generating {kernel_type} kernel: {e}")
            return ""

    def _create_sfpu_chain_compute_prompt(self, operation: str, op_config: Dict[str, str]) -> str:
        """Create specialized prompt for SFPU chain compute kernels"""

        if operation == "diode_equation":
            return f"""Generate a TT-Metal compute kernel for the diode equation: {op_config['formula']}

Requirements:
- Implement the diode current equation using SFPU operations
- Initialize all required operations before using them (pattern: operation_init() then operation())
- Use single-core pattern with standard circular buffers
- Follow TT-Metal API patterns from the provided examples

Filename: kernels/compute/{op_config['kernel_name']}.cpp
Generate only the .cpp code."""

        elif operation == "softplus":
            return f"""Generate a TT-Metal compute kernel for softplus: {op_config['formula']}

Requirements:
- Implement the softplus function using SFPU operations
- Initialize all required operations before using them (pattern: operation_init() then operation())
- Use single-core pattern with standard circular buffers
- Follow TT-Metal API patterns from the provided examples

Filename: kernels/compute/{op_config['kernel_name']}.cpp
Generate only the .cpp code."""

        else:
            return f"""Generate a TT-Metal compute kernel for {op_config['description']}.

This implements an SFPU operation chain. Follow the pattern:
1. Initialize all operations: {', '.join([f"{api}()" for api in op_config['api_init']])}
2. Load data to registers once
3. Chain operations: {' → '.join(op_config['api_compute'])}
4. Store result once

Filename: kernels/compute/{op_config['kernel_name']}.cpp
Generate only the .cpp code."""

    def _save_generated_files(
        self,
        output_dir: Path,
        operation: str,
        core_mode: str,
        kernels: Dict[str, str],
        host_code: str,
        cmake_content: str,
        refinement_logs: List[str],
    ):
        """Save all generated files to output directory"""

        # Create directory structure
        (output_dir / "kernels" / "compute").mkdir(parents=True, exist_ok=True)
        (output_dir / "kernels" / "dataflow").mkdir(parents=True, exist_ok=True)

        op_config = OPERATIONS[operation]

        # Save kernel files
        compute_file = output_dir / "kernels" / "compute" / f"{op_config['kernel_name']}.cpp"
        compute_file.write_text(kernels["compute"])

        if core_mode == "single":
            reader_file = output_dir / "kernels" / "dataflow" / "reader_binary_1_tile.cpp"
            writer_file = output_dir / "kernels" / "dataflow" / "writer_1_tile.cpp"
        else:
            reader_file = output_dir / "kernels" / "dataflow" / "reader_binary_tiles_partitioned.cpp"
            writer_file = output_dir / "kernels" / "dataflow" / "writer_tiles_partitioned.cpp"

        reader_file.write_text(kernels["reader"])
        writer_file.write_text(kernels["writer"])

        # Save host code and CMakeLists.txt if generated
        if host_code:
            project_name = f"{operation}_{core_mode}_example"
            host_file = output_dir / f"{project_name}.cpp"
            host_file.write_text(host_code)

        if cmake_content:
            cmake_file = output_dir / "CMakeLists.txt"
            cmake_file.write_text(cmake_content)

        # Save generation log
        log_content = f"""# TT-Metal Kernel Generation Log

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Operation:** {operation}
**Core Mode:** {core_mode}
**Model:** {self.model}

## Generated Files:
- {compute_file.name}
- {reader_file.name}
- {writer_file.name}
"""

        if host_code:
            log_content += f"- {project_name}.cpp\n"
        if cmake_content:
            log_content += f"- CMakeLists.txt\n"

        if refinement_logs:
            log_content += f"\n## Refinement Log:\n"
            for log_entry in refinement_logs:
                log_content += f"- {log_entry}\n"

        log_file = output_dir / "GENERATION_LOG.md"
        log_file.write_text(log_content)

        logger.info(f"Saved all files to {output_dir}")

    def _validate_kernels(self, output_dir: Path, operation: str, core_mode: str) -> Dict[str, any]:
        """Basic validation of generated kernels"""
        # TODO: Implement actual validation
        # This would involve compiling and running the kernels
        return {"validation": "not_implemented"}

    def _save_prompt_debug(
        self, kernel_type: str, operation: str, core_mode: str, system_prompt: str, user_prompt: str
    ):
        """Save the complete prompt (system + user) to a markdown file for debugging."""
        try:
            # Use the stored output directory if available, otherwise current directory
            if hasattr(self, "_current_output_dir") and self._current_output_dir:
                debug_dir = self._current_output_dir
                debug_dir.mkdir(parents=True, exist_ok=True)
            else:
                debug_dir = Path(".")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file = debug_dir / f"prompt_debug_{kernel_type}_{operation}_{core_mode}_{timestamp}.md"

            with open(debug_file, "w") as f:
                f.write(f"# TT-Metal Kernel Generator - Debug Prompt\n\n")
                f.write(f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Kernel Type:** {kernel_type}\n")
                f.write(f"**Operation:** {operation}\n")
                f.write(f"**Core Mode:** {core_mode}\n")
                f.write(f"**Model:** {self.model}\n\n")

                f.write("## System Prompt (RAG Context)\n\n")
                f.write("```\n")
                f.write(system_prompt)
                f.write("\n```\n\n")

                f.write("## User Prompt\n\n")
                f.write("```\n")
                f.write(user_prompt)
                f.write("\n```\n\n")

                # Add some statistics
                system_len = len(system_prompt)
                user_len = len(user_prompt)
                total_len = system_len + user_len

                f.write("## Prompt Statistics\n\n")
                f.write(f"- System Prompt Length: {system_len:,} characters\n")
                f.write(f"- User Prompt Length: {user_len:,} characters\n")
                f.write(f"- Total Prompt Length: {total_len:,} characters\n")
                f.write(f"- Estimated Tokens: ~{total_len // 4:,} tokens\n\n")

                # Extract RAG sources if available
                if "# SFPU_CHAIN EXAMPLE:" in system_prompt or "# SINGLE_CORE EXAMPLE:" in system_prompt:
                    f.write("## RAG Sources Included\n\n")
                    if "# SFPU_CHAIN EXAMPLE:" in system_prompt:
                        f.write("- SFPU Chain examples\n")
                    if "# SINGLE_CORE EXAMPLE:" in system_prompt:
                        f.write("- Single-core examples\n")
                    if "# MULTI_CORE EXAMPLE:" in system_prompt:
                        f.write("- Multi-core examples\n")

            logger.info(f"Saved prompt debug to: {debug_file}")

        except Exception as e:
            logger.warning(f"Failed to save prompt debug: {e}")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate TT-Metal kernels with iterative refinement",
        epilog="Example: %(prog)s --operation subtract --core-mode multi --refine --generate-host",
    )

    parser.add_argument(
        "--operation",
        choices=list(OPERATIONS.keys()),
        help="Operation to generate kernels for (required unless using --refine-target)",
    )

    parser.add_argument(
        "--core-mode",
        choices=list(CORE_MODES.keys()),
        help="Core mode: single or multi (required unless using --refine-target)",
    )

    parser.add_argument("--output", type=str, help="Output directory (default: auto-generated based on operation)")

    parser.add_argument(
        "--model", type=str, default=OPENAI_MODEL_DEFAULT, help=f"OpenAI model (default: {OPENAI_MODEL_DEFAULT})"
    )

    parser.add_argument("--refine", action="store_true", help="Enable iterative refinement with compilation feedback")

    parser.add_argument(
        "--refine-target",
        type=str,
        help="Path to existing programming example to refine (e.g., 'diode_equation_single_core_generated')",
    )

    parser.add_argument("--generate-host", action="store_true", help="Generate host code and CMakeLists.txt")

    parser.add_argument("--validate", action="store_true", help="Run validation tests on generated kernels")

    parser.add_argument(
        "--max-iterations", type=int, help=f"Maximum refinement iterations (default: {MAX_REFINEMENT_ITERATIONS})"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate argument combinations
    if args.refine_target:
        if not args.refine_target:
            logger.error("--refine-target requires a path to an existing example")
            return 1
    else:
        if not args.operation or not args.core_mode:
            logger.error("--operation and --core-mode are required unless using --refine-target")
            return 1

    # Check for API key (Groq or OpenAI)
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY or OPENAI_API_KEY environment variable not set")
        return 1

    # Check for mutually exclusive modes
    if args.refine_target and (args.operation or args.core_mode):
        logger.warning("When using --refine-target, operation and core-mode are auto-detected")

    # Display configuration
    max_iterations = args.max_iterations if args.max_iterations else MAX_REFINEMENT_ITERATIONS
    print("=" * 80)
    print("TT-Metal Kernel Generator with Iterative Refinement")
    print("=" * 80)

    # Different configuration display for refine-target mode
    if args.refine_target:
        print(f"Mode: Refining existing example")
        print(f"Target: {args.refine_target}")
        print(f"Model: {args.model}")
        print(f"Max Iterations: {max_iterations}")
        print(f"Validation: {'Enabled' if args.validate else 'Disabled'}")
    else:
        print(f"Operation: {OPERATIONS[args.operation]['description']}")
        print(f"Core Mode: {CORE_MODES[args.core_mode]['description']}")
        print(f"Model: {args.model}")
        print(f"Refinement: {'Enabled' if args.refine else 'Disabled'}")
        if args.refine:
            print(f"Max Iterations: {max_iterations}")
        print(f"Host Code: {'Generate' if args.generate_host else 'Skip'}")
        print(f"Validation: {'Enabled' if args.validate else 'Disabled'}")

    # Initialize generator
    generator = KernelGenerator(api_key, args.model)

    # Handle different modes
    try:
        if args.refine_target:
            # Refine existing example mode
            result = generator.refine_existing_example(
                target_path=args.refine_target,
                max_iterations=max_iterations,
            )
        else:
            # Normal generation mode
            output_dir = None
            if args.output:
                output_dir = Path(args.output)

            result = generator.generate_kernels(
                operation=args.operation,
                core_mode=args.core_mode,
                output_dir=output_dir,
                enable_refinement=args.refine,
                generate_host=args.generate_host,
                validate=args.validate,
                max_iterations=args.max_iterations,
            )

        if result["success"]:
            print("\n" + "=" * 80)
            print("✓ GENERATION SUCCESSFUL")
            print("=" * 80)
            print(f"Output Directory: {result['output_dir']}")

            if result["refinement_logs"]:
                print(f"Refinement Iterations: {len(result['refinement_logs'])}")

            print("\nGenerated Files:")
            for file_path in Path(result["output_dir"]).rglob("*"):
                if file_path.is_file():
                    print(f"  • {file_path.relative_to(result['output_dir'])}")
        else:
            print(f"\n✗ GENERATION FAILED: {result.get('error', 'Unknown error')}")
            return 1

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return 1

    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
