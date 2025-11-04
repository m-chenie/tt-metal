# TT-Metal Kernel Generation Log

**Date:** 2025-11-03 16:16:43
**Operation:** addition (C = A + B)
**Core Mode:** single-core implementation
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
4. If errors, run: ./generate_kernel.py --refine /home/m48chen/tt-metal/tt_metal/programming_examples/add_single_core_generated

---

## COMPUTE Kernel Prompt

```
Generate a TT-Metal compute kernel for addition (C = A + B) (C = A + B).

Requirements:
- Find the addition (C = A + B) API functions in the knowledge base
- Follow the single-core addition compute kernel pattern
- Use circular buffers: CB_0 (input A), CB_1 (input B), CB_16 (output)
- Single tile processing (1 tile at a time)
- Include comments explaining each step
- Filename: kernels/compute/add_2_tiles.cpp

Generate only the .cpp code.
```

---

## READER Kernel Prompt

```
Generate a TT-Metal reader kernel for single-core addition (C = A + B).

Requirements:
- Follow the single-core addition reader kernel pattern
- Read two tiles from DRAM to CB_0 and CB_1
- Use NOC async reads with barriers
- Runtime args: arg 0 = src0 address, arg 1 = src1 address
- Filename: kernels/dataflow/reader_binary_1_tile.cpp

Generate only the .cpp code.
```

---

## WRITER Kernel Prompt

```
Generate a TT-Metal writer kernel for single-core addition (C = A + B).

Requirements:
- Follow the single-core addition writer kernel pattern
- Write one tile from CB_16 to DRAM
- Use NOC async write with barrier
- Runtime arg: arg 0 = dst address
- Filename: kernels/dataflow/writer_1_tile.cpp

Generate only the .cpp code.
```

---
