#!/usr/bin/env python3
"""
RAG Knowledge Base Builder for TT-Metal Kernel Generator
Loads and processes existing TT-Metal examples to create rich context for LLM generation.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
import logging
from config import TT_METAL_HOME, RAG_SOURCES

logger = logging.getLogger(__name__)


class RAGKnowledgeBase:
    """Builds and manages the RAG knowledge base for kernel generation"""

    def __init__(self):
        self.knowledge_base = {}
        self.api_documentation = {}

    def build_knowledge_base(self, core_mode: str = "single") -> Dict[str, str]:
        """Build complete knowledge base for the specified core mode"""
        logger.info(f"Building RAG knowledge base for {core_mode}-core mode")

        # Load API documentation
        self.api_documentation = self._load_api_documentation()

        # Load examples based on core mode
        if core_mode == "single":
            examples = self._load_single_core_examples()
        else:
            examples = self._load_multi_core_examples()

        # Combine everything
        knowledge_base = {
            **self.api_documentation,
            **examples,
        }

        logger.info(f"Knowledge base built with {len(knowledge_base)} components")
        return knowledge_base

    def _load_file(self, filepath: Path) -> str:
        """Load a file safely and return its contents"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            logger.debug(f"Loaded {filepath} ({len(content)} chars)")
            return content
        except Exception as e:
            logger.warning(f"Could not load {filepath}: {e}")
            return ""

    def _load_api_documentation(self) -> Dict[str, str]:
        """Load and process API header files"""
        api_docs = {}

        for header_path in RAG_SOURCES["api_headers"]:
            full_path = TT_METAL_HOME / header_path
            content = self._load_file(full_path)
            if content:
                # Extract key API functions
                api_docs[f"api_{full_path.stem}"] = self._extract_api_functions(content)

        # Create consolidated API documentation
        api_docs["api_consolidated"] = self._create_api_summary(api_docs)
        return api_docs

    def _extract_api_functions(self, content: str) -> str:
        """Extract key API function signatures and documentation"""
        # Extract function declarations with ALWI or inline
        patterns = [
            r"ALWI\s+void\s+\w+\([^)]+\)[^;]*;",  # ALWI functions
            r"inline\s+void\s+\w+\([^)]+\)[^{]*{[^}]*}",  # Inline functions
            r"void\s+(add|sub|mul|mm|matmul|cb_|tile_)\w*\([^)]+\)",  # Specific API functions
        ]

        extracted = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            extracted.extend(matches)

        return "\n".join(extracted[:20])  # Limit to first 20 matches

    def _create_api_summary(self, api_docs: Dict[str, str]) -> str:
        """Create a consolidated API summary"""
        summary = """## TT-Metal Kernel API Summary

### Element-wise Binary Operations:
- add_tiles_init(cb_in0, cb_in1) - Initialize addition
- add_tiles(cb_in0, cb_in1, itile0, itile1, idst) - Perform addition
- sub_tiles_init(cb_in0, cb_in1) - Initialize subtraction
- sub_tiles(cb_in0, cb_in1, itile0, itile1, idst) - Perform subtraction
- mul_tiles_init(cb_in0, cb_in1) - Initialize multiplication
- mul_tiles(cb_in0, cb_in1, itile0, itile1, idst) - Perform multiplication
- binary_op_init_common(cb_in0, cb_in1, cb_out) - Common binary op initialization

### Matrix Multiplication:
- mm_init() - Initialize matrix multiplication
- matmul_tiles(cb_a, cb_b, itile0, itile1, idst) - Perform matrix multiplication

### Tile Management:
- tile_regs_acquire() - Acquire destination registers
- tile_regs_commit() - Commit results to packer
- tile_regs_wait() - Wait for packer completion
- tile_regs_release() - Release registers
- pack_tile(dst_index, cb_out) - Pack tile to circular buffer

### Circular Buffer Operations:
- cb_wait_front(cb_id, num_tiles) - Wait for input data
- cb_reserve_back(cb_id, num_tiles) - Reserve output space
- cb_push_back(cb_id, num_tiles) - Push output data
- cb_pop_front(cb_id, num_tiles) - Pop input data

### Memory Operations:
- noc_async_read_tile(src_addr, cb_id, tile_id) - Async read tile
- noc_async_write_tile(tile_id, cb_id, dst_addr) - Async write tile
- noc_async_read_barrier() - Wait for reads to complete
- noc_async_write_barrier() - Wait for writes to complete
"""
        return summary

    def _load_single_core_examples(self) -> Dict[str, str]:
        """Load single-core example implementations"""
        examples = {}

        for example_name in RAG_SOURCES["single_core_examples"]:
            example_path = TT_METAL_HOME / "tt_metal" / "programming_examples" / example_name

            # Load kernels
            compute_path = example_path / "kernels" / "compute"
            dataflow_path = example_path / "kernels" / "dataflow"

            # Find compute kernel
            for kernel_file in compute_path.glob("*.cpp"):
                examples[f"{example_name}_compute"] = self._load_file(kernel_file)
                break

            # Find reader kernel
            for kernel_file in dataflow_path.glob("reader*.cpp"):
                examples[f"{example_name}_reader"] = self._load_file(kernel_file)
                break

            # Find writer kernel
            for kernel_file in dataflow_path.glob("writer*.cpp"):
                examples[f"{example_name}_writer"] = self._load_file(kernel_file)
                break

            # Load host code if available
            host_files = list(example_path.glob("*.cpp"))
            if host_files:
                examples[f"{example_name}_host"] = self._load_file(host_files[0])

        return examples

    def _load_multi_core_examples(self) -> Dict[str, str]:
        """Load multi-core example implementations"""
        examples = {}

        for example_name in RAG_SOURCES["multi_core_examples"]:
            example_path = TT_METAL_HOME / "tt_metal" / "programming_examples" / example_name

            # Load kernels
            compute_path = example_path / "kernels" / "compute"
            dataflow_path = example_path / "kernels" / "dataflow"

            # Find compute kernel
            for kernel_file in compute_path.glob("*.cpp"):
                examples[f"{example_name}_compute"] = self._load_file(kernel_file)
                break

            # Find reader kernel
            for kernel_file in dataflow_path.glob("reader*.cpp"):
                examples[f"{example_name}_reader"] = self._load_file(kernel_file)
                break

            # Find writer kernel
            for kernel_file in dataflow_path.glob("writer*.cpp"):
                examples[f"{example_name}_writer"] = self._load_file(kernel_file)
                break

            # Load host code
            host_files = list(example_path.glob("*.cpp"))
            if host_files:
                examples[f"{example_name}_host"] = self._load_file(host_files[0])

            # Load CMakeLists.txt if available
            cmake_file = example_path / "CMakeLists.txt"
            if cmake_file.exists():
                examples[f"{example_name}_cmake"] = self._load_file(cmake_file)

        return examples

    def get_system_prompt(self, knowledge_base: Dict[str, str], core_mode: str, operation: str) -> str:
        """Create a comprehensive system prompt with RAG context"""

        # Select primary example based on core mode and operation
        if core_mode == "single":
            if operation in ["add", "subtract", "multiply"]:
                primary_example = "add_2_integers_in_compute"
            else:
                primary_example = "add_2_integers_in_compute"
        else:  # multi-core
            if operation == "matmul":
                primary_example = "matmul/matmul_multi_core"
            elif operation in ["add", "subtract", "multiply"]:
                primary_example = "subtract_multi_core_kernels_llm"
            else:
                primary_example = "matmul/matmul_multi_core"

        system_prompt = f"""You are an expert TT-Metal kernel developer for Tenstorrent's Wormhole architecture.

# KNOWLEDGE BASE (RAG Context)

## API Documentation
{knowledge_base.get('api_consolidated', '')}

## Primary Reference Example: {primary_example}

### Compute Kernel:
```cpp
{knowledge_base.get(f'{primary_example}_compute', 'Not available')}
```

### Reader Kernel:
```cpp
{knowledge_base.get(f'{primary_example}_reader', 'Not available')}
```

### Writer Kernel:
```cpp
{knowledge_base.get(f'{primary_example}_writer', 'Not available')}
```

### Host Code:
```cpp
{knowledge_base.get(f'{primary_example}_host', 'Not available')}
```

## Key Patterns for {core_mode.upper()}-Core Implementation:

### For Single-Core:
- Simple sequential processing (one tile at a time)
- Single reader/writer kernels
- Basic circular buffer management
- Straightforward host code with single core grid

### For Multi-Core:
- SPMD (Single Program Multiple Data) parallelization
- Work distribution using split_work_to_cores()
- Each core processes multiple tiles
- Partitioned reader/writer kernels
- Complex host code with mesh device management
- Use InterleavedAddrGenFast<true> for address generation

# YOUR ROLE

Generate high-quality TT-Metal kernels by:
1. Following the architectural patterns from the reference examples
2. Selecting correct API functions for the requested operation
3. Implementing proper error handling and optimization
4. Using modern TT-Metal best practices
5. Creating production-ready, well-documented code

When generating code:
- Include proper headers (#include <cstdint> for kernels)
- Add meaningful comments explaining the logic
- Use consistent variable naming and formatting
- Handle edge cases and error conditions
- Follow the exact patterns from working examples
"""

        return system_prompt
