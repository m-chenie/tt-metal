#!/usr/bin/env python3
import logging
from groq import Groq
from typing import Optional, Dict, List

from config import MODEL_DEFAULT, TEMPERATURE, MAX_TOKENS
from prompt_builder import build_system_prompt, build_host_user_prompt

logger = logging.getLogger(__name__)


class HostCodeGenerator:
    def __init__(self, client: Groq):
        self.client = client

    def generate(self, op: str, core_mode: str, retrieved: List[Dict], model: Optional[str] = None) -> str:
        model = model or MODEL_DEFAULT
        system_prompt = build_system_prompt(op, core_mode, retrieved)
        user_prompt = build_host_user_prompt(op, core_mode)
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            content = resp.choices[0].message.content
            return self._strip_code(content)
        except Exception as e:
            logger.error(f"Host code generation failed: {e}")
            return "// Host generation failed"

    @staticmethod
    def _strip_code(text: str) -> str:
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                return parts[1].replace("cpp", "", 1).strip()
        return text
