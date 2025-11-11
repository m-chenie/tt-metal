#!/usr/bin/env python3
"""
Diode Equation Generator for TT-Metal
Generates the diode equation: I = isat × (exp(V/vj) - 1) using SFPU chaining
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any
from generate_kernel import KernelGenerator
from config import TT_METAL_HOME

logger = logging.getLogger(__name__)


def create_diode_equation_prompt(operation_config: Dict[str, Any]) -> str:
    """Create specialized prompt for diode equation generation"""

    return f"""
# DIODE EQUATION KERNEL GENERATION

## Equation: {operation_config['formula']}
## Inputs: {', '.join(operation_config['inputs'])}

Generate a complete TT-Metal programming example that implements the diode current equation:

**I = isat × (exp(V/vj) - 1)**

Where:
- **V**: Forward voltage (input tensor)
- **vj**: Junction voltage (input tensor)
- **isat**: Saturation current (input tensor)
- **I**: Diode current (output tensor)

## Required Implementation:

### 1. SFPU Chain Operations (in sequence):
1. **Division**: `V/vj` (voltage ratio)
2. **Exponential**: `exp(V/vj)`
3. **Subtraction**: `exp(V/vj) - 1`
4. **Multiplication**: `isat × (exp(V/vj) - 1)` (final result)

### 2. Kernel Structure:
- **Reader Kernel**: Read V, vj, isat tensors + create constants (1.0)
- **Compute Kernel**: Chain the 4 SFPU operations keeping results in registers
- **Writer Kernel**: Write final I tensor

### 3. Host Code Requirements:
- Initialize 3 input tensors (V, vj, isat) with realistic diode parameters
- Set up 4 circular buffers (3 inputs + 1 output)
- Use 32x32 tiles with bfloat16 data format
- Include validation against CPU reference implementation

### 4. Validation Values:
Use these realistic diode parameters for testing:
- V: 0.1V to 0.8V (forward bias range)
- vj: 0.026V (thermal voltage at room temperature)
- isat: 1e-12A to 1e-9A (typical saturation current range)

### 5. Directory Structure:
Create complete project at: `tt_metal/programming_examples/diode_equation/`

## Key Implementation Notes:

### Compute Kernel Chain Pattern:
```cpp
// Initialize all SFPU operations
div_binary_tile_init();    // For V/vj
exp_tile_init();          // For exp(V/vj)
sub_binary_tile_init();   // For exp(V/vj) - 1
mul_binary_tile_init();   // For isat × result

// Load all inputs to registers
tile_regs_acquire();
copy_tile(cb_v, 0, 0);      // V -> R0
copy_tile(cb_vj, 0, 1);     // vj -> R1
copy_tile(cb_isat, 0, 2);   // isat -> R2
copy_tile(cb_ones, 0, 3);   // 1.0 -> R3

// Chain operations (keep in registers)
div_binary_tile(0, 1, 0);   // R0 = V/vj
exp_tile(0);                // R0 = exp(V/vj)
sub_binary_tile(0, 3, 0);   // R0 = exp(V/vj) - 1
mul_binary_tile(2, 0, 0);   // R0 = isat × (exp(V/vj) - 1)

// Write final result
pack_tile(0, cb_output);
tile_regs_release();
```

Generate production-ready TT-Metal code that demonstrates efficient SFPU chaining for this complex mathematical equation.
"""


def generate_diode_equation_example():
    """Generate the diode equation programming example"""

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key'")
        return False

    # Initialize generator
    generator = KernelGenerator(api_key)

    # Generate diode equation example
    print("🧪 Generating Diode Equation Programming Example...")
    print("📐 Formula: I = isat × (exp(V/vj) - 1)")
    print("⚡ Using SFPU chain operations for efficiency")

    try:
        result = generator.generate_kernels(
            operation="diode_equation", core_mode="single", enable_refinement=True, generate_host=True, validate=True
        )

        if result["success"]:
            output_dir = result.get("output_directory")
            print(f"✅ Diode equation example generated successfully!")
            print(f"📁 Location: {output_dir}")
            print(f"🔍 Files created:")
            for file in output_dir.rglob("*"):
                if file.is_file():
                    print(f"   - {file.relative_to(output_dir)}")
        else:
            print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Generation failed with exception: {e}")
        return False

    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = generate_diode_equation_example()
    exit(0 if success else 1)
