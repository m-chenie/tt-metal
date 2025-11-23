import json
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from config import INDEX_PATH, TOP_K, TT_METAL_HOME, CANONICAL_EXAMPLES, OPERATIONS


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def load_index() -> List[Dict[str, Any]]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Index not found at {INDEX_PATH}. Run ingest.py --reindex first.")
    with open(INDEX_PATH, "r") as f:
        return json.load(f)


def _load_canonical_example(rel_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a canonical example file directly.
    """
    try:
        full_path = TT_METAL_HOME / rel_path
        if not full_path.exists():
            return None

        text = full_path.read_text(encoding="utf-8", errors="ignore")
        tokens = _tokenize(text)
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        # Infer meta from path
        from ingest import _infer_meta

        meta = _infer_meta(full_path)

        return {
            "id": str(full_path),
            "path": str(full_path),
            "chunk": text,
            "tokens": tokens,
            "tf": tf,
            "meta": meta,
        }
    except Exception as e:
        print(f"Warning: Could not load canonical example {rel_path}: {e}")
        return None


def bm25(
    query: str, docs: List[Dict[str, Any]], k1: float = 1.6, b: float = 0.75
) -> List[Tuple[float, Dict[str, Any]]]:
    tokens = _tokenize(query)
    if not tokens:
        return []
    df = {}
    N = len(docs)
    avgdl = sum(len(d["tokens"]) for d in docs) / max(1, N)
    for doc in docs:
        seen = set()
        for t in doc["tokens"]:
            if t not in seen:
                df[t] = df.get(t, 0) + 1
                seen.add(t)

    scores = []
    for doc in docs:
        score = 0.0
        dl = len(doc["tokens"]) or 1
        tf_counts = doc["tf"]
        for t in tokens:
            tf = tf_counts.get(t, 0)
            if tf == 0:
                continue
            n_qi = df.get(t, 0)
            if n_qi == 0:
                continue
            idf = math.log((N - n_qi + 0.5) / (n_qi + 0.5) + 1)
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            score += idf * (tf * (k1 + 1) / denom)
        if score > 0:
            scores.append((score, doc))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:TOP_K]


def retrieve(query: str, filters: Dict[str, str] = None) -> List[Dict[str, Any]]:
    """
    Simple retrieve (legacy compatibility).
    """
    filters = filters or {}
    docs = load_index()

    def match(doc: Dict[str, Any]) -> bool:
        for key, val in filters.items():
            if val and doc["meta"].get(key) != val:
                return False
        return True

    filtered = [d for d in docs if match(d)]
    scored = bm25(query, filtered)
    return [doc for _, doc in scored]


def retrieve_smart(query: str, operation: str, core_mode: str = "", kernel_type: str = "") -> List[Dict[str, Any]]:
    """
    Smart multi-pass retrieval with canonical examples and deduplication.
    """
    docs = load_index()
    results = []
    seen_paths = set()

    # Determine operation type for canonical lookup
    op_type = OPERATIONS.get(operation, {}).get("operation_type", "binary")

    # Pass 1: Force-load canonical example for this operation type and kernel type
    canonical_map = CANONICAL_EXAMPLES.get(op_type, {})
    if kernel_type and kernel_type in canonical_map:
        canonical_path = canonical_map[kernel_type]
        canonical = _load_canonical_example(canonical_path)
        if canonical:
            results.append(canonical)
            seen_paths.add(canonical["path"])

    # Pass 1b: If operation has multiple inputs, also include eltwise_binary example
    # to show the multi-input TensorAccessor pattern (for reader kernels)
    op_info = OPERATIONS.get(operation, {})
    num_inputs = len(op_info.get("inputs", [])) + len(op_info.get("constants", []))
    if num_inputs > 1 and op_type != "binary" and kernel_type == "reader":
        binary_map = CANONICAL_EXAMPLES.get("binary", {})
        if kernel_type in binary_map:
            binary_path = binary_map[kernel_type]
            if binary_path not in seen_paths:
                binary_example = _load_canonical_example(binary_path)
                if binary_example:
                    results.append(binary_example)
                    seen_paths.add(binary_example["path"])

    # Pass 2: BM25 with exact filters (operation_type + kernel_type)
    exact_filters = {"operation": op_type}
    if kernel_type:
        exact_filters["kind"] = kernel_type
    if core_mode:
        exact_filters["core_mode"] = core_mode

    exact_matches = [
        d for d in docs if all(d["meta"].get(k) == v for k, v in exact_filters.items()) and d["path"] not in seen_paths
    ]

    if exact_matches:
        scored = bm25(query, exact_matches)
        for score, doc in scored[:3]:
            if doc["path"] not in seen_paths:
                results.append(doc)
                seen_paths.add(doc["path"])

    # Pass 3: Broaden to same operation type (ignore kernel_type)
    if len(results) < 4:
        broad_matches = [d for d in docs if d["meta"].get("operation") == op_type and d["path"] not in seen_paths]

        if broad_matches:
            scored = bm25(query, broad_matches)
            for score, doc in scored[:2]:
                if doc["path"] not in seen_paths:
                    results.append(doc)
                    seen_paths.add(doc["path"])

    # Pass 4: If still not enough, get similar kernel types from any operation
    if len(results) < TOP_K and kernel_type:
        kind_matches = [d for d in docs if d["meta"].get("kind") == kernel_type and d["path"] not in seen_paths]

        if kind_matches:
            scored = bm25(query, kind_matches)
            for score, doc in scored[:2]:
                if doc["path"] not in seen_paths:
                    results.append(doc)
                    seen_paths.add(doc["path"])

    return results[:TOP_K]


def retrieve_host_examples(operation: str) -> List[Dict[str, Any]]:
    """
    Retrieve complete host code examples for a given operation.

    Returns canonical host examples first, then searches for similar host code.
    """
    results = []
    seen_paths = set()

    # Determine operation type for canonical lookup
    op_type = OPERATIONS.get(operation, {}).get("operation_type", "binary")

    # Load canonical host example for this operation type
    canonical_map = CANONICAL_EXAMPLES.get(op_type, {})
    if "host" in canonical_map:
        canonical_path = canonical_map["host"]
        canonical = _load_canonical_example(canonical_path)
        if canonical:
            results.append(canonical)
            seen_paths.add(canonical["path"])

    # If operation has multiple inputs, also include eltwise_binary example
    # to show the multi-input TensorAccessor pattern
    op_info = OPERATIONS.get(operation, {})
    num_inputs = len(op_info.get("inputs", [])) + len(op_info.get("constants", []))
    if num_inputs > 1 and op_type != "binary":  # Don't duplicate if already binary
        binary_map = CANONICAL_EXAMPLES.get("binary", {})
        if "host" in binary_map:
            binary_path = binary_map["host"]
            if binary_path not in seen_paths:
                binary_example = _load_canonical_example(binary_path)
                if binary_example:
                    results.append(binary_example)
                    seen_paths.add(binary_example["path"])

    # Search for other host examples from similar operations
    docs = load_index()
    host_docs = [d for d in docs if d["meta"].get("kind") == "host" and d["path"] not in seen_paths]

    if host_docs:
        # Prioritize same operation type
        query = f"{operation} {op_type} host main distributed MeshDevice"
        scored = bm25(query, host_docs)

        for score, doc in scored[:2]:  # Top 2 additional host examples
            if doc["path"] not in seen_paths:
                results.append(doc)
                seen_paths.add(doc["path"])

    return results


def retrieve_cmake_examples(operation: str) -> List[Dict[str, Any]]:
    """
    Retrieve CMakeLists.txt examples from canonical examples.

    Returns CMakeLists.txt files from the same directories as canonical host examples.
    """
    results = []

    # Determine operation type for canonical lookup
    op_type = OPERATIONS.get(operation, {}).get("operation_type", "binary")

    # Load CMakeLists.txt from canonical example directory
    canonical_map = CANONICAL_EXAMPLES.get(op_type, {})
    if "host" in canonical_map:
        host_rel_path = canonical_map["host"]  # Already relative to TT_METAL_HOME
        # CMakeLists.txt should be in the same directory as the host file
        cmake_rel_path = str(Path(host_rel_path).parent / "CMakeLists.txt")

        cmake_doc = _load_canonical_example(cmake_rel_path)
        if cmake_doc:
            results.append(cmake_doc)

    # Also load from other operation types as additional examples
    for other_op_type, examples in CANONICAL_EXAMPLES.items():
        if other_op_type != op_type and "host" in examples:
            host_rel_path = examples["host"]  # Already relative to TT_METAL_HOME
            cmake_rel_path = str(Path(host_rel_path).parent / "CMakeLists.txt")

            cmake_doc = _load_canonical_example(cmake_rel_path)
            if cmake_doc and len(results) < 2:  # Limit to 2 examples
                results.append(cmake_doc)

    return results
