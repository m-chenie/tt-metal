#!/usr/bin/env python3
"""
TT-Metal Kernel Generator (Clean Version)
Generates only kernels using RAG. User handles host code and CMakeLists.txt manually.
If compilation fails, use --refine flag with path to fix issues.
"""

import os
import sys
import argparse
import re
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Supported operations
OPERATIONS = {
    "subtract": {
        "description": "subtraction (C = A - B)",
        "kernel_name": "subtract_2_tiles",
        "operation_symbol": "-",
    },
    "multiply": {
        "description": "multiplication (C = A * B)",
        "kernel_name": "multiply_2_tiles",
        "operation_symbol": "*",
    },
    "add": {
        "description": "addition (C = A + B)",
        "kernel_name": "add_2_tiles",
        "operation_symbol": "+",
    },
}

# Core modes
CORE_MODES = {
    "single": {
        "description": "single-core implementation",
        "suffix": "_single_core",
    },
    "multi": {
        "description": "multi-core implementation",
        "suffix": "_multi_core",
    },
}


class KernelGenerator:
    """Main kernel generator with iterative refinement"""

    def __init__(self, openai_api_key: str, model: str = OPENAI_MODEL_DEFAULT):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.rag_kb = RAGKnowledgeBase()
        self.refinement_engine = RefinementEngine(self.client)
        self.host_generator = HostCodeGenerator(self.client)

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

        # Build RAG knowledge base
        logger.info("Building RAG knowledge base...")
        knowledge_base = self.rag_kb.build_knowledge_base(core_mode)
        system_prompt = self.rag_kb.get_system_prompt(knowledge_base, core_mode, operation)

        # Generate initial kernels
        logger.info("Generating initial kernels...")
        kernels = self._generate_initial_kernels(operation, core_mode, system_prompt)

        if not kernels:
            logger.error("Failed to generate initial kernels")
            return {"success": False, "error": "Kernel generation failed"}

        # Generate host code if requested
        host_code = ""
        cmake_content = ""
        if generate_host:
            logger.info("Generating host code...")
            host_code = self.host_generator.generate_host_code(operation, core_mode, kernels, system_prompt, self.model)
            cmake_content = self.host_generator.generate_cmake_file(operation, core_mode)

        # Iterative refinement if enabled
        refinement_logs = []
        if enable_refinement and generate_host:
            logger.info("Starting iterative refinement...")
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

    def _generate_initial_kernels(self, operation: str, core_mode: str, system_prompt: str) -> Dict[str, str]:
        """Generate the initial set of kernels"""

        kernels = {}
        op_config = OPERATIONS[operation]

        # Generate compute kernel
        kernels["compute"] = self._generate_single_kernel("compute", operation, core_mode, system_prompt, op_config)

        # Generate reader kernel
        kernels["reader"] = self._generate_single_kernel("reader", operation, core_mode, system_prompt, op_config)

        # Generate writer kernel
        kernels["writer"] = self._generate_single_kernel("writer", operation, core_mode, system_prompt, op_config)

        # Check if all kernels were generated successfully
        if not all(kernels.values()):
            logger.error("Failed to generate one or more kernels")
            return {}

        return kernels

    def _generate_single_kernel(
        self, kernel_type: str, operation: str, core_mode: str, system_prompt: str, op_config: Dict[str, str]
    ) -> str:
        """Generate a single kernel using the LLM"""

        # Create specific prompt for this kernel type
        if kernel_type == "compute":
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


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate TT-Metal kernels with iterative refinement",
        epilog="Example: %(prog)s --operation subtract --core-mode multi --refine --generate-host",
    )

    parser.add_argument(
        "--operation", choices=list(OPERATIONS.keys()), required=True, help="Operation to generate kernels for"
    )

    parser.add_argument(
        "--core-mode", choices=list(CORE_MODES.keys()), required=True, help="Core mode: single or multi"
    )

    parser.add_argument("--output", type=str, help="Output directory (default: auto-generated based on operation)")

    parser.add_argument(
        "--model", type=str, default=OPENAI_MODEL_DEFAULT, help=f"OpenAI model (default: {OPENAI_MODEL_DEFAULT})"
    )

    parser.add_argument("--refine", action="store_true", help="Enable iterative refinement with compilation feedback")

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

    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return 1

    # Display configuration
    max_iterations = args.max_iterations if args.max_iterations else MAX_REFINEMENT_ITERATIONS
    print("=" * 80)
    print("TT-Metal Kernel Generator with Iterative Refinement")
    print("=" * 80)
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

    # Set output directory
    output_dir = None
    if args.output:
        output_dir = Path(args.output)

    # Generate kernels
    try:
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
