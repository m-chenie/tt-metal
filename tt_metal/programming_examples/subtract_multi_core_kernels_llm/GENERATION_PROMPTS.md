# Kernel Generation Log

**Date:** 2025-11-03 14:21:21
**Operation:** subtraction (C = A - B)
**Core Mode:** multi-core implementation
**Model:** gpt-4o

## Note on RAG
The system prompt contains the complete knowledge base (API docs + multi-core implementation-core examples).
User prompts below are minimal - they just reference the RAG context.

---

## COMPUTE Kernel

**User Prompt:**
```
Generate a TT-Metal multi-core compute kernel for subtraction (C = A - B) (C = A - B).

Requirements:
- Look in the knowledge base and find the subtraction (C = A - B) API functions (sub_tiles_init, sub_tiles, etc.)
- Follow the WORKING multi-core subtract compute kernel pattern EXACTLY
- Use circular buffers: CB_0 (input A), CB_1 (input B), CB_16 (output)
- Handle multiple tiles per core (work distribution)
- Use runtime args to get work assignment: arg(0) = num_tiles
- Include #include <cstdint> and proper API headers
- Filename: kernels/compute/subtract_2_tiles.cpp

Generate only the .cpp code.
```

---

## READER Kernel

**User Prompt:**
```
Generate a TT-Metal multi-core reader kernel for subtraction (C = A - B).

Requirements:
- Follow the WORKING multi-core subtract reader kernel pattern EXACTLY
- Use InterleavedAddrGenFast<true> for address generation (NOT TensorAccessor)
- Each core reads its assigned tiles from DRAM to CB_0 and CB_1
- Runtime args: src0_addr, src1_addr, Mt, Kt, Nt, start_tile_id, num_tiles
- Use NOC async reads with barriers: noc_async_read_tile, noc_async_read_barrier
- Include only: #include <cstdint>
- For element-wise: read same tile_id from both src0 and src1
- Filename: kernels/dataflow/reader_binary_tiles_partitioned.cpp

Generate only the .cpp code.
```

---

## WRITER Kernel

**User Prompt:**
```
Generate a TT-Metal multi-core writer kernel for subtraction (C = A - B).

Requirements:
- Follow the WORKING multi-core subtract writer kernel pattern EXACTLY
- Use InterleavedAddrGenFast<true> for address generation (NOT TensorAccessor)
- Each core writes its assigned result tiles from CB_16 to DRAM
- Runtime args: dst_addr, start_tile_id, num_tiles
- Use NOC async writes with barriers: noc_async_write_tile, noc_async_write_barrier
- Include only: #include <cstdint>
- Filename: kernels/dataflow/writer_tiles_partitioned.cpp

Generate only the .cpp code.
```

---
