# Bitmap Optimization for 100K Permissions

## Overview
This PR optimizes Milvus's bitmap query module to handle large-scale permission sets (up to 100K bits) more efficiently. The goal is to reduce memory usage by ~3× and improve CPU computation efficiency by 2-3×.

## Current Implementation Analysis
The existing implementation already uses Roaring Bitmap with three modes:
1. ROARING: For high-cardinality datasets
2. BITSET: For low-cardinality datasets (< 500 entries)
3. MMAP: Memory-mapped optimization for large datasets

## Changes Made

### 1. Large Bitmap Detection
- Added `LARGE_BITMAP_THRESHOLD = 100000` constant
- Modified `ChooseIndexLoadMode` to force ROARING mode for large permission sets
- This ensures optimal compression for large permission bitmaps

### 2. Permission Binning (BitmapBinning.h)
- Implemented binning strategy to segment 100K permissions into smaller groups
- Default bin size of 1000 permissions per bin
- Reduces active bits per query by focusing on relevant permission segments
- Improves cache locality and reduces memory pressure

### 3. Order-preserving Bin-based Clustering (OrBiC.h)
- Added clustering optimization for permission bits
- Orders permissions by cardinality to optimize query patterns
- Minimizes candidate checks during bitmap operations
- Reduces I/O overhead for large permission sets

### 4. Performance Testing
Added comprehensive test suite:
1. Unit Tests (test_array_bitmap_index.cpp):
   - Large permission set handling
   - Memory usage verification
   - Query performance validation

2. Benchmarks (bitmap_benchmark.cpp):
   - Range query performance
   - Bitmap operations efficiency
   - Memory consumption metrics

### 5. Build System Updates
- Modified CMakeLists.txt to support direct build without Conan
- Added OpenMP and Threads dependencies for parallel processing

## Expected Improvements
1. Memory Usage:
   - ~3× reduction through Roaring Bitmap compression
   - Additional savings from binning optimization

2. CPU Efficiency:
   - 2-3× improvement in query performance
   - Better cache utilization through binning
   - Reduced computation through clustered access patterns

3. Query Latency:
   - Faster range queries using bin-based access
   - Improved performance for permission checks
   - Reduced I/O overhead through clustering

## Testing Instructions
1. Build the project:
```bash
cd internal/core
mkdir -p build && cd build
cmake -DUSE_CONAN=OFF ..
make -j$(nproc)
```

2. Run tests:
```bash
./milvus_core_unittest
./bitmap_benchmark
```

## Implementation Notes
- The implementation preserves backward compatibility
- No changes to the public API
- Automatic mode selection based on permission set size
- Optional advanced optimizations through binning and clustering
