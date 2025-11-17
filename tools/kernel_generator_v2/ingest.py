#!/usr/bin/env python3
"""
Build a lightweight BM25 index over programming examples (compute/dataflow/host).
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import re

from config import (
    PROGRAMMING_EXAMPLES_DIR,
    API_HEADER_DIRS,
    INDEX_PATH,
    SKIP_DIRS,
    CODE_EXTS,
    DOC_EXTS,
    MAX_FILE_SIZE,
)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _should_skip_file(path: Path) -> bool:
    """
    Filter out generated files, logs, and other non-example content.
    """
    path_str = str(path)
    name = path.name

    # Skip generated directories and files
    if "_generated" in path_str or "GENERATION_LOG" in name:
        return True

    # Skip build artifacts and temp files
    if any(skip in path.parts for skip in SKIP_DIRS):
        return True

    # Skip overly large files (likely not kernel examples)
    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return True
    except:
        return True

    return False


def _infer_meta(path: Path) -> Dict[str, str]:
    """
    Infer metadata from file path with improved accuracy.
    """
    meta = {"operation": "", "core_mode": "", "kind": ""}
    parts = [p.lower() for p in path.parts]
    path_str = str(path).lower()
    name = path.name.lower()

    # Core mode detection - be specific about directory structure
    if "single_core" in path_str or "single-core" in path_str:
        meta["core_mode"] = "single"
    elif "multi_core" in path_str or "multicore" in path_str or "multi-core" in path_str:
        meta["core_mode"] = "multi"

    # Operation type detection - prioritize specific patterns
    if "sfpu_eltwise_chain" in path_str or "custom_sfpu" in path_str:
        meta["operation"] = "sfpu_chain"
    elif "diode" in path_str or "softplus" in path_str:
        meta["operation"] = "sfpu_chain"
    elif "eltwise_binary" in path_str:
        meta["operation"] = "binary"
    elif "add_2_integers_in_compute" in path_str:
        meta["operation"] = "binary"
    elif "matmul" in path_str:
        meta["operation"] = "matmul"
    elif "vecadd" in path_str:
        meta["operation"] = "binary"
    elif "multiply" in path_str:
        meta["operation"] = "binary"

    # API header tagging
    if path.suffix in {".h", ".hpp"}:
        if "compute_kernel_api" in path_str:
            meta["operation"] = meta["operation"] or "api_compute"
        if "tt-metalium" in path_str or "host_api" in path_str:
            meta["operation"] = meta["operation"] or "api_host"

    # Kernel type detection - be granular
    if "compute" in parts:
        meta["kind"] = "compute"
    elif "dataflow" in parts:
        # Distinguish reader vs writer
        if "reader" in name or "read" in name:
            meta["kind"] = "reader"
        elif "writer" in name or "write" in name:
            meta["kind"] = "writer"
        else:
            meta["kind"] = "dataflow"
    elif path.suffix in {".md", ".rst", ".txt"}:
        meta["kind"] = "doc"
    elif path.suffix in CODE_EXTS:
        meta["kind"] = "host"
    else:
        meta["kind"] = "other"

    return meta


def walk_sources() -> List[Dict]:
    """
    Walk source directories and index complete files (no chunking).
    """
    documents = []
    roots = [PROGRAMMING_EXAMPLES_DIR] + API_HEADER_DIRS

    for root in roots:
        for path in root.rglob("*"):
            # Skip non-files
            if not path.is_file():
                continue

            # Skip unwanted files
            if _should_skip_file(path):
                continue

            # Only process code and doc files
            if path.suffix not in CODE_EXTS | DOC_EXTS:
                continue

            # Read file content
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"Warning: Could not read {path}: {e}")
                continue

            # Skip empty or trivial files
            if len(text.strip()) < 50:
                continue

            # Tokenize and compute term frequencies
            tokens = _tokenize(text)
            if not tokens:
                continue

            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            # Infer metadata
            meta = _infer_meta(path)

            # Store complete file as single document
            documents.append(
                {
                    "id": str(path),
                    "path": str(path),
                    "chunk": text,  # Complete file, not chunked
                    "tokens": tokens,
                    "tf": tf,
                    "meta": meta,
                }
            )

    return documents


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index for kernel generator v2.")
    parser.add_argument("--reindex", action="store_true", help="Rebuild index even if it exists.")
    args = parser.parse_args()

    if INDEX_PATH.exists() and not args.reindex:
        print(f"Index already exists at {INDEX_PATH}. Use --reindex to rebuild.")
        return 0

    documents = walk_sources()
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w") as f:
        json.dump(documents, f)
    print(f"Indexed {len(documents)} chunks into {INDEX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
