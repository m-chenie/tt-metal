#!/usr/bin/env python3
import logging
from groq import Groq
from typing import Optional, Dict, List

from config import MODEL_DEFAULT, TEMPERATURE, MAX_TOKENS
from prompt_builder import build_host_system_prompt, build_host_user_prompt
from retriever import retrieve_host_examples

logger = logging.getLogger(__name__)


class HostCodeGenerator:
    def __init__(self, client: Groq):
        self.client = client

    def generate(
        self, op: str, core_mode: str, retrieved: List[Dict] = None, model: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate host code and CMakeLists.txt using enhanced prompting with:
        - Canonical host template
        - Complete host examples
        - Host API signatures
        - CMakeLists.txt examples

        Returns:
            Dict with keys 'host_code' and 'cmake'
        """
        model = model or MODEL_DEFAULT

        # Retrieve complete host code examples (ignores 'retrieved' kernels)
        host_examples = retrieve_host_examples(op)

        # Build specialized host prompts
        system_prompt = build_host_system_prompt(op, core_mode, host_examples)
        user_prompt = build_host_user_prompt(op, core_mode)

        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            content = resp.choices[0].message.content
            return self._extract_host_and_cmake(content, op, core_mode)
        except Exception as e:
            logger.error(f"Host code generation failed: {e}")
            return {"host_code": "// Host generation failed", "cmake": self._generate_fallback_cmake(op, core_mode)}

    @staticmethod
    def _extract_host_and_cmake(text: str, op: str, core_mode: str) -> Dict[str, str]:
        """Extract both host code and CMakeLists.txt from LLM response."""
        result = {"host_code": "", "cmake": ""}

        if "```" not in text:
            result["host_code"] = text
            result["cmake"] = HostCodeGenerator._generate_fallback_cmake(op, core_mode)
            return result

        # Split by code fences
        blocks = text.split("```")

        for i in range(1, len(blocks), 2):  # Odd indices are code blocks
            block = blocks[i]
            # Strip language tag
            lines = block.split("\n", 1)
            if len(lines) > 1:
                lang_tag = lines[0].strip().lower()
                code = lines[1]
            else:
                lang_tag = ""
                code = block

            # Identify block type
            if "cmake" in lang_tag or "cmake_minimum_required" in code[:100].lower():
                result["cmake"] = code.strip()
            elif "cpp" in lang_tag or "#include" in code[:200]:
                result["host_code"] = code.strip()
            elif not result["host_code"]:  # First block defaults to host code
                result["host_code"] = code.strip()

        # Fallback if CMakeLists.txt wasn't generated
        if not result["cmake"]:
            logger.warning("LLM did not generate CMakeLists.txt, using fallback template")
            result["cmake"] = HostCodeGenerator._generate_fallback_cmake(op, core_mode)

        return result

    @staticmethod
    def _generate_fallback_cmake(op: str, core_mode: str) -> str:
        """Generate CMakeLists.txt from template if LLM fails."""
        from config import CMAKELISTS_TEMPLATE

        project_name = f"{op}_{core_mode}_v2"
        source_file = f"{op}_{core_mode}_v2.cpp"

        return CMAKELISTS_TEMPLATE.format(project_name=project_name, source_file=source_file)
