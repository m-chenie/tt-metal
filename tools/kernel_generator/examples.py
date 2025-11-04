#!/usr/bin/env python3
"""
Example usage and test script for TT-Metal Kernel Generator
Demonstrates various generation modes and validates the results.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from generate_kernel import KernelGenerator
from validator import validate_generated_project


def run_example(operation: str, core_mode: str, enable_refinement: bool = True, generate_host: bool = True):
    """Run a single example generation"""

    print(f"\n{'='*60}")
    print(f"Generating {operation} {core_mode}-core kernels")
    print(f"Refinement: {'Enabled' if enable_refinement else 'Disabled'}")
    print(f"Host Code: {'Generate' if generate_host else 'Skip'}")
    print(f"{'='*60}")

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not set")
        return False

    # Initialize generator
    generator = KernelGenerator(api_key)

    # Generate
    try:
        result = generator.generate_kernels(
            operation=operation,
            core_mode=core_mode,
            enable_refinement=enable_refinement,
            generate_host=generate_host,
            validate=False,  # We'll validate manually
        )

        if not result["success"]:
            print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
            return False

        print(f"✅ Generation successful!")
        print(f"   Output: {result['output_dir']}")

        if result["refinement_logs"]:
            print(f"   Refinement iterations: {len(result['refinement_logs'])}")

        # Validate the result
        if generate_host:
            print("\n🔍 Validating generated project...")
            validation_results = validate_generated_project(Path(result["output_dir"]), operation, core_mode)

            print(f"   File structure: {'✅' if validation_results['file_structure']['success'] else '❌'}")
            print(f"   Kernel syntax: {'✅' if validation_results['kernel_syntax']['success'] else '❌'}")
            print(f"   Compilation: {'✅' if validation_results['compilation']['success'] else '❌'}")

            if not validation_results["overall_success"]:
                print("   ⚠️  Some validation checks failed")
                for error in validation_results.get("compilation", {}).get("errors", []):
                    print(f"      - {error}")
            else:
                print("   🎉 All validation checks passed!")

        return result["success"]

    except Exception as e:
        print(f"❌ Exception during generation: {e}")
        return False


def main():
    """Run example generations"""

    print("🚀 TT-Metal Kernel Generator Examples")
    print("=====================================")

    # List of examples to run
    examples = [
        # Start with simpler single-core examples
        ("add", "single", False, True),  # Single-core addition without refinement
        ("subtract", "single", True, True),  # Single-core subtraction with refinement
        # Then try multi-core examples
        ("multiply", "multi", True, True),  # Multi-core multiplication with refinement
        ("add", "multi", True, True),  # Multi-core addition with refinement
    ]

    results = []

    for operation, core_mode, refinement, host_code in examples:
        success = run_example(operation, core_mode, refinement, host_code)
        results.append((operation, core_mode, success))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    success_count = sum(1 for _, _, success in results if success)
    total_count = len(results)

    for operation, core_mode, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {operation} ({core_mode}-core)")

    print(f"\nSuccess rate: {success_count}/{total_count}")

    if success_count == total_count:
        print("🎉 All examples completed successfully!")
        return 0
    else:
        print("⚠️  Some examples failed. Check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
