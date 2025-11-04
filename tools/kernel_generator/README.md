# TT-Metal Kernel Generator

An AI-powered tool for generating TT-Metal kernels with iterative refinement and automatic host code generation.

## Features

- **RAG-based Knowledge Base**: Uses existing TT-Metal examples as context
- **Iterative Refinement**: Automatically compiles and fixes compilation errors

## Quick Start

```bash
# Generate single-core addition kernels
python generate_kernel.py --operation add --core-mode single

# Generate multi-core multiplication with iterative refinement
python generate_kernel.py --operation multiply --core-mode multi --refine

# Generate with host code and validation
python generate_kernel.py --operation subtract --core-mode multi --generate-host --validate
```

## Directory Structure

```
kernel_generator/
├── generate_kernel.py          # Main generator script
├── rag_knowledge_base.py       # RAG context builder
├── kernel_templates.py         # Kernel generation templates
├── host_code_generator.py      # Host code generation
├── refinement_engine.py        # Iterative refinement system
├── validator.py                # Kernel validation pipeline
├── config.py                   # Configuration settings
└── examples/                   # Generated examples output
```

## How It Works

1. **RAG Context Building**: Loads and processes existing TT-Metal examples
2. **Template-based Generation**: Uses LLM with rich context to generate kernels
3. **Compilation Feedback**: Attempts to compile and captures errors
4. **Iterative Refinement**: Feeds compilation errors back to LLM for fixes
5. **Host Code Generation**: Creates complete runnable examples
6. **Validation**: Runs generated examples to verify correctness
