#!/usr/bin/env python3
"""
RAG Knowledge Base for TT-Metal Kernel Generator
Builds knowledge base from configured TT-Metal examples and provides intelligent retrieval
"""

import os
import logging
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

# Import configuration
from config import RAG_SOURCES, EXAMPLES_DIR, TT_METAL_HOME


class RAGKnowledgeBase:
    """RAG knowledge base that builds from configured TT-Metal examples"""

    def __init__(self):
        self.api_headers = ""  # Always included
        self.host_api_headers = ""  # Always included
        self.programming_examples = defaultdict(list)  # Selectively included
        self._built = False

    def build_knowledge_base(self, core_mode: str = "single", operation_type: str = "standard") -> Dict[str, str]:
        """Build knowledge base from configured TT-Metal examples"""
        if self._built:
            return {}

        logger.info("Building knowledge base from configured TT-Metal sources...")

        # Always extract all API headers (included in every prompt)
        self._extract_api_headers()
        self._extract_host_api_headers()

        # Extract programming examples by category (selectively included)
        self._extract_programming_examples()

        self._built = True
        logger.info("Knowledge base built successfully")
        return {}

    def get_system_prompt_smart(self, query: str, core_mode: str = "single") -> str:
        """Create system prompt with smart retrieval based on query"""
        if not self._built:
            self.build_knowledge_base(core_mode)

        # Always include: system message + all API headers + all host API headers
        context_parts = [
            "You are an expert TT-Metal kernel developer for Tenstorrent's Wormhole architecture.",
            self.api_headers,
            self.host_api_headers,
        ]

        # Selectively include programming examples based on query
        relevant_examples = self._select_relevant_examples(query, core_mode)
        if relevant_examples:
            context_parts.extend(relevant_examples)

        context_parts.append("Generate high-quality, production-ready TT-Metal code following the patterns above.")

        return "\n\n".join(context_parts)

    def _extract_api_headers(self):
        """Extract all compute API headers (always included in prompts)"""
        api_content = []
        api_content.append("# TT-METAL COMPUTE API")

        for header_path in RAG_SOURCES["api_headers"]:
            full_path = TT_METAL_HOME / header_path
            if full_path.exists():
                content = self._extract_header_apis(full_path)
                if content:
                    api_content.append(f"\n## {full_path.name}:")
                    api_content.append(content)
            else:
                logger.warning(f"API header not found: {full_path}")

        self.api_headers = "\n".join(api_content)

    def _extract_host_api_headers(self):
        """Extract all host API headers (always included in prompts)"""
        host_content = []
        host_content.append("# TT-METAL HOST API")

        for header_path in RAG_SOURCES["host_api_headers"]:
            full_path = TT_METAL_HOME / header_path
            if full_path.exists():
                content = self._extract_header_apis(full_path)
                if content:
                    host_content.append(f"\n## {full_path.name}:")
                    host_content.append(content)
            else:
                logger.warning(f"Host API header not found: {full_path}")

        self.host_api_headers = "\n".join(host_content)

    def _extract_programming_examples(self):
        """Extract programming examples by category"""
        # Single core examples
        for example_name in RAG_SOURCES["single_core_examples"]:
            example_path = EXAMPLES_DIR / example_name
            if example_path.exists():
                pattern = self._extract_example_pattern(example_path, "single_core")
                if pattern:
                    self.programming_examples["single_core"].append(pattern)
            else:
                logger.warning(f"Single core example not found: {example_path}")

        # SFPU chain examples
        for example_name in RAG_SOURCES["sfpu_chain_examples"]:
            example_path = EXAMPLES_DIR / example_name
            if example_path.exists():
                pattern = self._extract_example_pattern(example_path, "sfpu_chain")
                if pattern:
                    self.programming_examples["sfpu_chain"].append(pattern)
            else:
                logger.warning(f"SFPU chain example not found: {example_path}")

        # Multi core examples
        for example_name in RAG_SOURCES["multi_core_examples"]:
            example_path = EXAMPLES_DIR / example_name
            if example_path.exists():
                pattern = self._extract_example_pattern(example_path, "multi_core")
                if pattern:
                    self.programming_examples["multi_core"].append(pattern)
            else:
                logger.warning(f"Multi core example not found: {example_path}")

    def _extract_header_apis(self, header_path: Path) -> str:
        """Extract API functions from header file"""
        content = self._read_file_safe(header_path)
        if not content:
            return ""

        # Extract function declarations, defines, and important patterns
        api_lines = []
        lines = content.split("\n")

        for line in lines:
            stripped = line.strip()
            # Skip comments and empty lines
            if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
                continue

            # Extract function declarations, defines, and important patterns
            if any(
                pattern in stripped
                for pattern in [
                    "ALWAYSINLINE",
                    "inline",
                    "constexpr",
                    "FORCE_INLINE",
                    "cb_wait_front",
                    "cb_reserve_back",
                    "cb_push_back",
                    "cb_pop_front",
                    "tile_regs_",
                    "pack_tile",
                    "copy_tile",
                    "unpack_tilize",
                    "noc_async_read",
                    "noc_async_write",
                    "_barrier",
                    "_init(",
                    "_tile(",
                    "_tiles(",
                    "get_compile_time_arg",
                    "#define",
                    "enum class",
                    "namespace",
                ]
            ):
                api_lines.append(stripped)

        return "\n".join(api_lines[:50]) if api_lines else ""  # Limit to prevent huge context

    def _extract_example_pattern(self, example_path: Path, category: str) -> str:
        """Extract key patterns from a programming example"""
        pattern_parts = [f"# {category.upper()} EXAMPLE: {example_path.name}"]

        # Look for kernels directory
        kernels_dir = example_path / "kernels"
        if kernels_dir.exists():
            # Extract compute kernel patterns
            compute_dir = kernels_dir / "compute"
            if compute_dir.exists():
                for cpp_file in compute_dir.glob("*.cpp"):
                    content = self._extract_key_code_patterns(cpp_file, "compute")
                    if content:
                        pattern_parts.append(f"\n## Compute Kernel ({cpp_file.name}):")
                        pattern_parts.append(f"```cpp\n{content}\n```")

            # Extract dataflow kernel patterns
            dataflow_dir = kernels_dir / "dataflow"
            if dataflow_dir.exists():
                for cpp_file in dataflow_dir.glob("*.cpp"):
                    content = self._extract_key_code_patterns(cpp_file, "dataflow")
                    if content:
                        kernel_type = "reader" if "reader" in cpp_file.name else "writer"
                        pattern_parts.append(f"\n## {kernel_type.title()} Kernel ({cpp_file.name}):")
                        pattern_parts.append(f"```cpp\n{content}\n```")

        # Extract host code pattern
        for cpp_file in example_path.glob("*.cpp"):
            if "kernel" not in cpp_file.name:  # Skip kernel files, get main host file
                content = self._extract_key_code_patterns(cpp_file, "host")
                if content:
                    pattern_parts.append(f"\n## Host Code ({cpp_file.name}):")
                    pattern_parts.append(f"```cpp\n{content}\n```")
                break  # Just get one main host file

        return "\n".join(pattern_parts) if len(pattern_parts) > 1 else ""

    def _extract_key_code_patterns(self, file_path: Path, code_type: str) -> str:
        """Extract key code patterns from source file"""
        content = self._read_file_safe(file_path)
        if not content:
            return ""

        lines = content.split("\n")
        key_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            # Keep important patterns based on code type
            important_keywords = []
            if code_type == "compute":
                important_keywords = [
                    "cb_wait_front",
                    "cb_reserve_back",
                    "cb_push_back",
                    "cb_pop_front",
                    "tile_regs_acquire",
                    "tile_regs_release",
                    "pack_tile",
                    "copy_tile",
                    "init_sfpu",
                    "_tile_init",
                    "_tile(",
                    "_binary_tile",
                    "void MAIN",
                    "#include",
                    "namespace NAMESPACE",
                ]
            elif code_type == "dataflow":
                important_keywords = [
                    "noc_async_read",
                    "noc_async_write",
                    "_barrier",
                    "get_arg_val",
                    "get_write_ptr",
                    "get_read_ptr",
                    "cb_reserve_back",
                    "cb_push_back",
                    "cb_wait_front",
                    "cb_pop_front",
                    "void kernel_main",
                    "#include",
                ]
            elif code_type == "host":
                important_keywords = [
                    "CreateProgram",
                    "CreateKernel",
                    "SetRuntimeArgs",
                    "EnqueueProgram",
                    "CreateBuffer",
                    "WriteBuffer",
                    "ReadBuffer",
                    "CreateDevice",
                    "ConfigureProgram",
                    "#include",
                    "int main",
                    "tt::tt_metal",
                ]

            if any(keyword in stripped for keyword in important_keywords):
                key_lines.append(line.rstrip())  # Keep original indentation

        # Limit to prevent huge context
        return "\n".join(key_lines[:30]) if key_lines else ""

    def _select_relevant_examples(self, query: str, core_mode: str) -> List[str]:
        """Select relevant programming examples based on query and core mode"""
        relevant = []
        q = query.lower()

        # Determine which example categories to include
        if "sfpu" in q or "chain" in q or "exp" in q or "log" in q or "diode" in q:
            # SFPU chain operations
            relevant.extend(self.programming_examples.get("sfpu_chain", []))
            # Also include single core for basic patterns
            relevant.extend(self.programming_examples.get("single_core", []))
        elif core_mode == "multi" or "multi" in q:
            # Multi-core specific
            relevant.extend(self.programming_examples.get("multi_core", []))
        else:
            # Default to single core
            relevant.extend(self.programming_examples.get("single_core", []))

        # Limit to prevent token overflow
        return relevant[:3]  # Max 3 examples

    def _read_file_safe(self, file_path: Path) -> str:
        """Safely read file contents"""
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return ""

    def get_system_prompt(self, knowledge_base: Dict[str, str], core_mode: str, operation: str) -> str:
        """Legacy compatibility method"""
        return self.get_system_prompt_smart(f"{operation} {core_mode}", core_mode)
