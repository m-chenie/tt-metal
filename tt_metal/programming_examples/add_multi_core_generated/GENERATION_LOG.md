# TT-Metal Kernel Generation Log

**Date:** 2025-11-03 20:44:57
**Operation:** addition (C = A + B)
**Core Mode:** multi-core implementation
**Model:** gpt-4o

## Workflow
1. ✅ Generated kernels only (no host code, no CMakeLists.txt)
2. ⏳ Manually create host code and CMakeLists.txt
3. ⏳ If compilation fails, use --refine flag

## Generated Kernels
- compute kernel: kernels/compute/*.cpp
- reader kernel: kernels/dataflow/reader_*.cpp
- writer kernel: kernels/dataflow/writer_*.cpp

## Next Steps
1. Create CMakeLists.txt (copy from similar example)
2. Create host code (copy from similar example and adapt)
3. Build with cmake and make
4. If errors, run: ./generate_kernel.py --refine /home/m48chen/tt-metal/tt_metal/programming_examples/add_multi_core_generated

---

## COMPUTE Kernel Prompt

```
Generate a TT-Metal multi-core compute kernel for addition (C = A + B) (C = A + B).

Requirements:
- Find the addition (C = A + B) API functions in the knowledge base
- Follow multi-core matmul compute kernel patterns for work distribution
- Use circular buffers: CB_0 (input A), CB_1 (input B), CB_16 (output)
- Handle multiple tiles per core (work distribution)
- Use runtime args to get work assignment
- Include proper headers and comments
- Filename: kernels/compute/add_2_tiles.cpp

Generate only the .cpp code.
```

---

## READER Kernel Prompt

```
Generate a TT-Metal multi-core reader kernel for addition (C = A + B).

Requirements:
- Follow multi-core matmul reader kernel patterns
- Each core reads its assigned tiles from DRAM to CB_0 and CB_1
- Use proper address generation and NOC async reads
- Runtime args for work distribution
- For element-wise ops: read same tile_id from both sources
- Filename: kernels/dataflow/reader_binary_tiles_partitioned.cpp

Generate only the .cpp code.
```

---

## WRITER Kernel Prompt

```
Generate a TT-Metal multi-core writer kernel for addition (C = A + B).

Requirements:
- Follow multi-core matmul writer kernel patterns
- Each core writes its assigned result tiles from CB_16 to DRAM
- Use proper address generation and NOC async writes
- Runtime args for work distribution
- Filename: kernels/dataflow/writer_tiles_partitioned.cpp

Generate only the .cpp code.
```

---
