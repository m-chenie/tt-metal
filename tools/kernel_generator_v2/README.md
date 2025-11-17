# TT-Metal Kernel Generator v2 (light RAG, no refinement)

An alternate generator that:
- Builds a lightweight BM25 index over `tt_metal/programming_examples` (compute/dataflow/host).
- Retrieves relevant snippets by op/core mode to condition the LLM.
- Generates kernels (compute + reader + writer) and optionally host code.
- Skips iterative refinement/compilation for fast turnaround.

## Quick start
```bash
cd tools/kernel_generator_v2
# Build the local index
python ingest.py --reindex

# Generate add single-core kernels
python generate_kernel.py --operation add --core-mode single

# Generate kernels + host code and save prompt debug
python generate_kernel.py --operation matmul --core-mode multi --generate-host --save-prompt
```

Environment:
- Set `GROQ_API_KEY` or `OPENAI_API_KEY`. Default model is `llama-3.3-70b-versatile` (see `config.py`).

Outputs:
- Generated dirs land in `tt_metal/programming_examples/{operation}{suffix}_v2_generated/`, with `kernels/compute`, `kernels/dataflow`, and optional host `.cpp` + `CMakeLists.txt`.
- If `--save-prompt` is used, the assembled prompt is saved as `prompt_debug.md` in the output dir.

## Files
```
kernel_generator_v2/
├── README.md
├── config.py            # Paths, ops/core definitions, model settings
├── ingest.py            # Build BM25 index from programming_examples
├── retriever.py         # BM25 retrieval and filtering
├── prompt_builder.py    # System/user prompt assembly
├── host_code_generator.py
├── generate_kernel.py   # CLI entry: retrieve + LLM generate
└── index.json           # Cached index (created by ingest.py)
```

## Notes
- Indexing uses simple per-chunk split; see `config.py` for chunk sizes and skip dirs.
- If examples change/add, rerun `python ingest.py --reindex`.
- This version omits refinement and build/validation; pair with your own compile/test loop.
