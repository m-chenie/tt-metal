#!/usr/bin/env python3
"""
Lightweight kernel generator (no refinement).
- Pulls context via BM25 over programming_examples.
- Builds a tight prompt and asks the LLM for kernels (and optional host code).
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict

from groq import Groq

from config import (
    OPERATIONS,
    CORE_MODES,
    PROGRAMMING_EXAMPLES_DIR,
    MODEL_DEFAULT,
    TEMPERATURE,
    MAX_TOKENS,
)
from retriever import retrieve, retrieve_smart
from prompt_builder import (
    build_system_prompt,
    build_kernel_user_prompt,
    build_host_system_prompt,
    build_host_user_prompt,
)
from host_code_generator import HostCodeGenerator

# Note: Canonical examples are now handled by retrieve_smart()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kernel_generator_v2")


def parse_args():
    ap = argparse.ArgumentParser(description="TT-Metal kernel generator v2.")

    # Mode selection: either generate OR iterate (mutually exclusive)
    mode_group = ap.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--operation",
        choices=list(OPERATIONS.keys()),
        help="Operation to generate (use with --core-mode for initial generation)",
    )
    mode_group.add_argument(
        "--iterate", action="store_true", help="Iterative refinement mode (requires --example-path)"
    )

    # Arguments for initial generation
    ap.add_argument("--core-mode", choices=list(CORE_MODES.keys()), help="Core mode (required with --operation)")
    ap.add_argument("--generate-host", action="store_true", help="Also generate host code")
    ap.add_argument("--output", type=str, help="Optional output dir (defaults under programming_examples)")

    # Arguments for iteration
    ap.add_argument("--example-path", type=str, help="Path to existing example (required with --iterate)")
    ap.add_argument("--max-iterations", type=int, default=5, help="Maximum refinement iterations (default: 5)")

    # Common arguments
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--save-prompt", action="store_true", help="Save assembled prompts to the output dir")

    args = ap.parse_args()

    # Validation
    if args.operation and not args.core_mode:
        ap.error("--operation requires --core-mode")
    if args.iterate and not args.example_path:
        ap.error("--iterate requires --example-path")
    if args.core_mode and not args.operation:
        ap.error("--core-mode requires --operation")

    return args


def split_blocks(text: str, op: str) -> Dict[str, str]:
    """Heuristic split of LLM output into compute/reader/writer blocks."""
    blocks = {"compute": "", "reader": "", "writer": ""}

    # Extract code blocks
    pieces = []
    if "```" in text:
        raw_parts = text.split("```")
        for i in range(1, len(raw_parts), 2):
            code = raw_parts[i]
            # Strip language tag if present
            if code.startswith("cpp\n") or code.startswith("cpp "):
                code = code[code.find("\n") + 1 :]
            elif code.startswith("c++\n") or code.startswith("c++ "):
                code = code[code.find("\n") + 1 :]
            elif code.startswith("c\n") or code.startswith("c "):
                code = code[code.find("\n") + 1 :]
            pieces.append(code)
    else:
        pieces = [text]

    compute_name = OPERATIONS[op]["compute_kernel"].replace(".cpp", "")

    # Try to identify each block by looking for distinctive markers
    for p in pieces:
        body = p.strip()
        if not body or len(body) < 50:  # Skip empty or trivial blocks
            continue

        lower = body.lower()
        first_lines = "\n".join(body.split("\n")[:5]).lower()  # Check first 5 lines for labels

        # Check for explicit labels in comments (e.g., // COMPUTE KERNEL:)
        has_compute_label = "compute kernel" in first_lines or f"// {compute_name}" in first_lines
        has_reader_label = "reader kernel" in first_lines or "// reader" in first_lines
        has_writer_label = "writer kernel" in first_lines or "// writer" in first_lines

        # Look for distinctive function signatures
        has_kernel_main = "void kernel_main()" in body
        has_main_func = "void MAIN" in body or "namespace NAMESPACE" in body

        # Look for distinctive API patterns
        has_noc_read = "noc_async_read" in body
        has_noc_write = "noc_async_write" in body
        has_sfpu = "init_sfpu" in body or "tile_regs_acquire" in body or "pack_tile" in body

        # Identify kernel type with priority to labels
        tag = ""
        if has_compute_label or (has_main_func and has_sfpu):
            tag = "compute"
        elif has_reader_label or (has_kernel_main and has_noc_read and not has_noc_write):
            tag = "reader"
        elif has_writer_label or (has_kernel_main and has_noc_write):
            tag = "writer"

        # Fallback: generic keyword matching
        if not tag and len(body) > 100:
            if "writer" in lower:
                tag = "writer"
            elif "reader" in lower:
                tag = "reader"
            elif "compute" in lower or compute_name in lower:
                tag = "compute"

        # Only assign if we don't already have this kernel (prefer first occurrence)
        if tag and not blocks[tag]:
            blocks[tag] = body
            logger.info(f"Identified {tag} kernel ({len(body)} chars)")

    # Log warnings for missing kernels
    for ktype, code in blocks.items():
        if not code:
            logger.warning(f"{ktype.title()} kernel not found in LLM response")

    return blocks


def write_kernels(output_dir: Path, op: str, core_mode: str, blocks: Dict[str, str]):
    op_cfg = OPERATIONS[op]
    compute_name = op_cfg["compute_kernel"]

    kernels_dir = output_dir / "kernels"
    (kernels_dir / "compute").mkdir(parents=True, exist_ok=True)
    (kernels_dir / "dataflow").mkdir(parents=True, exist_ok=True)

    if not blocks["compute"]:
        logger.warning("Compute kernel missing from model output.")
    (kernels_dir / "compute" / compute_name).write_text(blocks["compute"])

    if core_mode == "single":
        reader_name = "reader_binary_1_tile.cpp"
        writer_name = "writer_1_tile.cpp"
    else:
        reader_name = "reader_binary_tiles_partitioned.cpp"
        writer_name = "writer_tiles_partitioned.cpp"

    (kernels_dir / "dataflow" / reader_name).write_text(blocks.get("reader", ""))
    (kernels_dir / "dataflow" / writer_name).write_text(blocks.get("writer", ""))


def save_prompt(
    output_dir: Path, system_prompt: str, user_prompt: str, host_system_prompt: str = None, host_user_prompt: str = None
):
    """Save prompts to debug file. If host prompts are provided, append them as well."""
    content = f"# Prompt Debug\n\n## Kernel Generation\n\n### System\n```\n{system_prompt}\n```\n\n### User\n```\n{user_prompt}\n```\n"

    if host_system_prompt and host_user_prompt:
        content += f"\n\n## Host Code Generation\n\n### System\n```\n{host_system_prompt}\n```\n\n### User\n```\n{host_user_prompt}\n```\n"

    debug_path = output_dir / "prompt_debug.md"
    debug_path.write_text(content)


def main(selected_args=None):
    args = selected_args or parse_args()

    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY or OPENAI_API_KEY")

    client = Groq(api_key=api_key)

    # Route to iteration mode if --iterate flag is set
    if args.iterate:
        from iterative_refine import iterative_refine

        iterative_refine(client, args)
        return

    # Otherwise, do initial generation
    host_gen = HostCodeGenerator(client)

    logger.info("Retrieving context...")
    op_type = OPERATIONS[args.operation].get("operation_type", "")

    # Retrieve examples for each kernel type separately
    retrieved = []

    # Get compute kernel examples
    compute_query = f"{args.operation} {op_type} {args.core_mode} compute kernel sfpu chain"
    compute_examples = retrieve_smart(compute_query, args.operation, args.core_mode, "compute")
    retrieved.extend(compute_examples[:2])  # Top 2 compute examples

    # Get reader kernel examples
    reader_query = f"{args.operation} {op_type} {args.core_mode} reader dataflow noc_async_read"
    reader_examples = retrieve_smart(reader_query, args.operation, args.core_mode, "reader")
    retrieved.extend(reader_examples[:2])  # Top 2 reader examples

    # Get writer kernel examples
    writer_query = f"{args.operation} {op_type} {args.core_mode} writer dataflow noc_async_write"
    writer_examples = retrieve_smart(writer_query, args.operation, args.core_mode, "writer")
    retrieved.extend(writer_examples[:2])  # Top 2 writer examples

    # Get relevant API headers (NEW: retrieve API documentation)
    if op_type == "sfpu_chain":
        api_query = "exp div sub mul add log binary tile init SFPU eltwise"
    else:
        api_query = "add sub mul matmul tile binary eltwise"

    api_examples = retrieve(api_query, {"operation": "api_compute"})
    retrieved.extend(api_examples[:3])  # Top 3 API headers

    if not retrieved:
        logger.warning("No retrieved examples matched; proceeding with minimal context.")

    system_prompt = build_system_prompt(args.operation, args.core_mode, retrieved)
    user_prompt = build_kernel_user_prompt(args.operation, args.core_mode)

    logger.info("Calling LLM for kernels...")
    resp = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    content = resp.choices[0].message.content

    blocks = split_blocks(content, args.operation)

    if args.output:
        output_dir = Path(args.output)
    else:
        suffix = CORE_MODES[args.core_mode]["suffix"]
        output_dir = PROGRAMMING_EXAMPLES_DIR / f"{args.operation}{suffix}_v2_generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    write_kernels(output_dir, args.operation, args.core_mode, blocks)

    host_system_prompt = None
    host_user_prompt = None

    if args.generate_host:
        logger.info("Generating host code...")
        host_result = host_gen.generate(args.operation, args.core_mode, retrieved, model=args.model)

        # Capture the prompts used for host generation
        from retriever import retrieve_host_examples

        host_examples = retrieve_host_examples(args.operation)
        host_system_prompt = build_host_system_prompt(args.operation, args.core_mode, host_examples)
        host_user_prompt = build_host_user_prompt(args.operation, args.core_mode)

        # Write host code
        host_path = output_dir / f"{args.operation}_{args.core_mode}_v2.cpp"
        host_path.write_text(host_result["host_code"])

        # Write CMakeLists.txt (from LLM or fallback template)
        cmake_path = output_dir / "CMakeLists.txt"
        cmake_path.write_text(host_result["cmake"])

    # Always save original prompts (even without --save-prompt) for iteration to use
    original_prompt_file = output_dir / "original_generation_prompt.md"
    original_prompt_file.write_text(
        f"# Original Generation Prompt\n\n## Kernel Generation\n\n### System\n```\n{system_prompt}\n```\n\n### User\n```\n{user_prompt}\n```\n"
        + (
            f"\n\n## Host Code Generation\n\n### System\n```\n{host_system_prompt}\n```\n\n### User\n```\n{host_user_prompt}\n```\n"
            if host_system_prompt and host_user_prompt
            else ""
        )
    )

    # Always save original prompts (even without --save-prompt) for iteration to use
    original_prompt_file = output_dir / "original_generation_prompt.md"
    original_prompt_file.write_text(
        f"# Original Generation Prompt\n\n## Kernel Generation\n\n### System\n```\n{system_prompt}\n```\n\n### User\n```\n{user_prompt}\n```\n"
        + (
            f"\n\n## Host Code Generation\n\n### System\n```\n{host_system_prompt}\n```\n\n### User\n```\n{host_user_prompt}\n```\n"
            if host_system_prompt and host_user_prompt
            else ""
        )
    )

    if args.save_prompt:
        save_prompt(output_dir, system_prompt, user_prompt, host_system_prompt, host_user_prompt)

    logger.info(f"Done. Output at {output_dir}")


if __name__ == "__main__":
    main()
