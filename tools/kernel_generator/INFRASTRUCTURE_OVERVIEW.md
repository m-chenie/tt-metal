# TT-Metal Kernel Generator Infrastructure

## Overview

I've created a comprehensive infrastructure for generating TT-Metal kernels with iterative refinement, moving beyond your original RAG-based approach to include:

1. **Iterative Refinement**: Automatically compiles generated kernels and uses compilation errors to improve the code
2. **Host Code Generation**: Creates complete runnable examples with CMakeLists.txt
3. **Validation Pipeline**: Tests file structure, syntax, compilation, and basic execution
4. **Integration with tt-metal repo**: Properly integrated into the tt-metal repository structure

## Architecture

```
tt-metal/tools/kernel_generator/
├── generate_kernel.py          # Main generation script with refinement
├── rag_knowledge_base.py       # Enhanced RAG system
├── refinement_engine.py        # Compilation feedback & iterative fixing
├── host_code_generator.py      # Complete host code generation
├── validator.py                # Kernel validation pipeline
├── config.py                   # Centralized configuration
├── examples.py                 # Usage examples and tests
├── setup.sh                    # Setup script
├── requirements.txt            # Python dependencies
└── generated_examples/         # Output directory
```

## Key Improvements Over Original

### 1. **Iterative Refinement Engine**
- Compiles generated kernels automatically
- Captures compilation errors and feeds them back to the LLM
- Iteratively fixes issues (up to 3 iterations)
- Uses structured error parsing and targeted fixes

### 2. **Complete Host Code Generation**
- Generates full host code with device setup
- Creates proper CMakeLists.txt files
- Handles both single-core and multi-core patterns
- Includes validation and performance timing

### 3. **Enhanced RAG Knowledge Base**
- More comprehensive API documentation
- Better example selection based on operation/mode
- Structured knowledge base with fallbacks
- Improved system prompts with clearer patterns

### 4. **Validation Pipeline**
- File structure validation
- Syntax checking for common issues
- Compilation testing with proper error reporting
- Basic execution validation

### 5. **Better Integration**
- Proper tt-metal repository integration
- Backward compatibility with existing scripts
- Centralized configuration management
- Comprehensive logging and error handling

## Usage Examples

### Basic Generation
```bash
cd tt-metal/tools/kernel_generator
./generate_kernel.py --operation add --core-mode single
```

### With Iterative Refinement
```bash
./generate_kernel.py --operation subtract --core-mode multi --refine --generate-host
```

### From KernelBench (backward compatibility)
```bash
cd KernelBench/scripts
./gpt4_kernel_generator_new.py --operation multiply --core-mode multi --refine --generate-host --validate
```

## Setup Instructions

1. **Install dependencies**:
   ```bash
   cd tt-metal/tools/kernel_generator
   ./setup.sh
   ```

2. **Set environment variables**:
   ```bash
   export GROQ_API_KEY="your-api-key-here"
   export TT_METAL_HOME="/path/to/tt-metal"  # Optional, auto-detected
   ```

3. **Test the installation**:
   ```bash
   cd KernelBench/scripts
   ./test_kernel_generator_integration.py
   ```

## What This Solves

### Multi-core Kernel Issues
- **Better Examples**: Uses working multi-core subtract example as primary reference
- **Iterative Fixing**: Compiles and fixes common multi-core issues automatically
- **Pattern Guidance**: Clearer system prompts with specific multi-core patterns
- **API Corrections**: Fixes common mistakes like using wrong address generators

### Host Code Generation
- **Complete Examples**: Generates full runnable projects, not just kernels
- **Proper Integration**: Uses correct TT-Metal APIs and patterns
- **Build System**: Creates proper CMakeLists.txt with all dependencies
- **Validation**: Includes golden reference and result checking

### Scalability and Maintenance
- **Modular Design**: Each component is separate and testable
- **Configuration Management**: Centralized settings for easy tuning
- **Extensibility**: Easy to add new operations or patterns
- **Integration**: Works with existing workflows while providing new capabilities

## Next Steps

1. **Test the infrastructure**:
   ```bash
   cd KernelBench/scripts
   ./test_kernel_generator_integration.py
   ```

2. **Generate some examples**:
   ```bash
   cd tt-metal/tools/kernel_generator
   python3 examples.py
   ```

3. **Try iterative refinement on your problematic multi-core cases**:
   ```bash
   ./generate_kernel.py --operation subtract --core-mode multi --refine --generate-host --validate
   ```

The key innovation here is the **iterative refinement loop** - instead of being "less specific" with prompts, we let the LLM make mistakes and then automatically feed compilation errors back to fix them. This approach should handle the complexity of multi-core kernels much better than trying to get everything right in the first generation pass.

Would you like me to test this infrastructure with your OPENAI_API_KEY, or would you prefer to try it out yourself first?
