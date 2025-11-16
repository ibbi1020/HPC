# KLT CUDA Optimization Checklist and Implementation Guide

## Priority Order (by Impact on Bottlenecks)
1. ✅ Texture memory for interpolation (40% bottleneck) [DONE]
2. ✅ Shared memory tiling for convolution (40% bottleneck combined) [DONE]
3. ✅ Constant memory for Gaussian kernels (supports convolution)
4. Coalesced memory access (memory-bound ops)
5. ✅ Persistent GPU buffers (overhead reduction) [DONE]
6. Async kernel launches and cudaMemCpyAsync (latency hiding) [partial - needs async version]
7. ✅ Block and grid size optimization (occupancy) [DONE]
8. Eigenvalue kernel optimization (5% bottleneck)
9. Additional memory transfer optimizations (communication)

---

## Optimization 1: Texture Memory for Interpolation

### Target Files
- `interpolate_cuda.cu`
- `interpolate_cuda.h`
- `trackFeatures.c` (caller)

### Current State
- Naive kernel `cudaNaiveInterpolate()` performs manual bilinear interpolation: `value = (1-ax)(1-ay)·p₀ + ax(1-ay)·p₁ + (1-ax)ay·p₂ + ax·ay·p₃`
- Each interpolation: 4 global memory reads, 8 FLOPs
- Called 2M+ times per tracking session (8,645 features × 49 window pixels + 6,235 gradient calls × 49)

### Implementation Strategy
1. **Create texture object management**:
   - Add `cudaTextureObject_t` to function signatures or pass as parameter
   - Bind `_KLT_FloatImage->data` to texture object with `cudaCreateTextureObject()`
   - Configure descriptor: `cudaFilterModeLinear` (hardware bilinear), `cudaAddressModeClamp` (boundary handling)
   - Unbind/destroy after use

2. **Modify kernel signature**:
   ```cuda
   __global__ void cudaTextureInterpolate(
       cudaTextureObject_t texObj,
       int width, int height,
       const float* coords_x, const float* coords_y,
       float* results, int numPoints
   ) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx >= numPoints) return;
       float x = coords_x[idx];
       float y = coords_y[idx];
       // Hardware bilinear interpolation
       results[idx] = tex2D<float>(texObj, x + 0.5f, y + 0.5f); // +0.5 for pixel center
   }
   ```

3. **Update caller in trackFeatures.c**:
   - Allocate texture once per pyramid level
   - Reuse for all features at that level
   - Pattern: `#ifdef USE_CUDA_TEXTURE_INTERPOLATION` around texture path

4. **Boundary handling**:
   - Set `cudaAddressModeClamp` to prevent OOB reads
   - Or add explicit bounds check: `if (x < 0 || x >= width-1 || y < 0 || y >= height-1) return 0.0f;`

5. **Expected impact**: 5-10x speedup on interpolation (40% → 4-8%), total ~30-35% overall speedup

### Verification
- Profile with `make ncu-quick-example3`
- Check "tex" unit utilization in ncu report
- Verify results match CPU within epsilon (floating point tolerance)

---

## Optimization 2: Shared Memory Tiling for Convolution [DONE]

### Target Files
- `convolve_gpu.cu`
- `convolve_gpu.h`
- `convolve.c` (caller)

### Implementation Status: ✅ COMPLETE
**Date Completed**: October 31, 2025  
**Approach**: Cooperative tile loading with halos, zero-padding boundary handling

### What Was Implemented

1. **New CUDA Kernels** (`convolve_gpu.cu`):
   - ✅ `convolveHorizShared()` - Horizontal convolution with shared memory
   - ✅ `convolveVertShared()` - Vertical convolution with shared memory
   - Configuration: 16×16 tiles, MAX_KERNEL_RADIUS=10

2. **Shared Memory Layout**:
   - **Horizontal**: `s_tile[TILE_HEIGHT][TILE_WIDTH + 2*MAX_KERNEL_RADIUS]` (16×36)
   - **Vertical**: `s_tile[TILE_HEIGHT + 2*MAX_KERNEL_RADIUS][TILE_WIDTH]` (36×16)
   - Memory per block: 2,304 bytes (well within 48KB limit)

3. **Cooperative Loading Pattern**:
   - Each thread loads multiple elements (ceil(36/16) = 3 loads)
   - Halos loaded in same pass as main tile
   - Zero-padding for out-of-bounds accesses
   - Single `__syncthreads()` after loading

4. **Host Wrappers** (`convolve_gpu.cu`):
   - ✅ `launchConvolveHorizShared()` - Fixed 16×16 block size
   - ✅ `launchConvolveVertShared()` - Fixed 16×16 block size

5. **Integration** (`convolve.c`):
   - ✅ Added `#ifdef USE_CUDA_CONVOLUTION_SHARED` conditionals
   - Falls back to original kernels if shared memory disabled
   - Both `_convolveImageHoriz()` and `_convolveImageVert()` updated

6. **Configuration** (`cuda_config.h`):
   - ✅ Added `USE_CUDA_CONVOLUTION_SHARED` switch

### Performance Impact
- **Horizontal convolution**: Expected 2-3× speedup (reduced redundant reads)
- **Vertical convolution**: Expected 5-10× speedup (eliminates strided access penalty)
- **Overall**: 3-6× speedup on convolution operations (40% of total runtime)
- **Cumulative with Opt 1+5+7**: **15-25× vs baseline CPU**

### Key Implementation Details
- **Boundary handling**: Zero-padding (matches CPU behavior)
- **Tile size**: 16×16 (256 threads, optimal for memory coalescing)
- **Halo size**: 20 elements (supports up to 21-element kernels)
- **Occupancy**: ~75% on Tesla T4 (multiple blocks per SM)
- **Bank conflicts**: Avoided by halo padding

### Original Strategy (for reference)
- Separable 1D convolution: `_convolveImageHoriz()` and `_convolveImageVert()`
- Horizontal pass: better locality (row-major)
- Vertical pass: poor locality (stride = width, ~30% bottleneck)
- Each pixel reloaded multiple times (once per kernel tap)

### Implementation Strategy
1. **2D thread block layout**:
   ```cuda
   #define TILE_WIDTH 16
   #define TILE_HEIGHT 16
   #define KERNEL_RADIUS 10  // For 21-element kernel
   
   __global__ void cudaConvolveHorizShared(
       const float* input, float* output,
       int width, int height,
       const float* kernel, int kernelWidth
   ) {
       __shared__ float tile[TILE_HEIGHT][TILE_WIDTH + 2*KERNEL_RADIUS];
       
       int tx = threadIdx.x, ty = threadIdx.y;
       int col = blockIdx.x * TILE_WIDTH + tx;
       int row = blockIdx.y * TILE_HEIGHT + ty;
   ```

2. **Load tile with halo (ghost cells)**:
   ```cuda
       // Main tile
       if (col < width && row < height) {
           tile[ty][tx + KERNEL_RADIUS] = input[row * width + col];
       }
       
       // Left halo
       if (tx < KERNEL_RADIUS && col >= tx) {
           tile[ty][tx] = input[row * width + (col - KERNEL_RADIUS)];
       }
       
       // Right halo
       if (tx < KERNEL_RADIUS && col + TILE_WIDTH < width) {
           tile[ty][tx + TILE_WIDTH + KERNEL_RADIUS] = 
               input[row * width + (col + TILE_WIDTH)];
       }
       
       __syncthreads(); // Wait for all loads
   ```

3. **Compute convolution from shared memory**:
   ```cuda
       if (col < width && row < height) {
           float sum = 0.0f;
           for (int k = 0; k < kernelWidth; k++) {
               sum += tile[ty][tx + k] * kernel[k];
           }
           output[row * width + col] = sum;
       }
   }
   ```

4. **Vertical convolution (similar pattern)**:
   - Tile layout: `tile[TILE_HEIGHT + 2*KERNEL_RADIUS][TILE_WIDTH]`
   - Load halos vertically (top/bottom)
   - Compute along columns

5. **Grid/block sizing**:
   ```cuda
   dim3 blockDim(TILE_WIDTH, TILE_HEIGHT);
   dim3 gridDim((width + TILE_WIDTH - 1) / TILE_WIDTH,
                (height + TILE_HEIGHT - 1) / TILE_HEIGHT);
   cudaConvolveHorizShared<<<gridDim, blockDim>>>(...)
   ```

6. **Expected impact**: 3-5x speedup on convolution (40% → 8-13%), total ~25-30% overall speedup

### Verification
- Check shared memory usage with ncu: `ncu --metrics shared_memory_utilization`
- Verify occupancy >50%
- Compare output with CPU version pixel-by-pixel

---

## Optimization 3: Constant Memory for Gaussian Kernels [DONE]

### Target Files
- `convolve_gpu.cu`
- `convolve_gpu.h`
- `convolve.c` (kernel generation)

### Implementation Status: ✅ COMPLETE
**Date Completed**: October 31, 2025  
**Approach**: Constant memory for kernel weights, shared memory for image data (complementary)

### What Was Implemented

1. **Constant Memory Declarations** (`convolve_gpu.cu`):
   - ✅ `__constant__ float d_gaussKernel[64]` - Smoothing kernel weights
   - ✅ `__constant__ float d_gaussDerivKernel[64]` - Derivative kernel weights  
   - ✅ `__constant__ int d_kernelWidth` - Active kernel size
   - ✅ `__constant__ int d_derivKernelWidth` - Derivative kernel size
   - Total: 512 bytes (well within 64KB limit)

2. **Kernel Upload Functions** (`convolve_gpu.cu`):
   - ✅ `cudaSetGaussianKernel()` - Uploads smoothing kernel to constant memory
   - ✅ `cudaSetGaussianDerivKernel()` - Uploads derivative kernel to constant memory
   - Uses `cudaMemcpyToSymbol()` for constant memory transfers

3. **New Kernel Variants** (`convolve_gpu.cu`):
   - ✅ `convolveHorizSharedConstant()` - Shared memory + constant memory
   - ✅ `convolveVertSharedConstant()` - Shared memory + constant memory
   - Read image data from shared memory (Opt 2 - unchanged)
   - Read kernel weights from constant memory (Opt 3 - new)

4. **Host Wrappers** (`convolve_gpu.cu`):
   - ✅ `launchConvolveHorizSharedConstant()` 
   - ✅ `launchConvolveVertSharedConstant()`
   - Simplified signatures (no kernel_data parameter)

5. **Integration** (`convolve.c`):
   - ✅ Added nested `#ifdef USE_CUDA_CONSTANT_MEMORY` conditionals
   - Uploads kernel before each convolution operation
   - Falls back to shared-only or naive versions if disabled
   - Both `_convolveImageHoriz()` and `_convolveImageVert()` updated

6. **Configuration** (`cuda_config.h`):
   - ✅ Added `USE_CUDA_CONSTANT_MEMORY` switch

### Performance Impact
- **Broadcast efficiency**: All threads read same kernel value → 1 transaction serves 32 threads
- **Dedicated cache**: 64KB constant cache, zero misses after warmup
- **Expected speedup**: 10-20% on convolution kernels
- **Overall impact**: 3-5% additional speedup on full KLT pipeline
- **Cumulative with Opt 1+2+5+7**: **18-30× vs baseline CPU**

### Key Implementation Details
- **Complementary design**: Image data uses shared memory, kernel weights use constant memory
- **No interference**: Separate memory spaces, independent caches
- **Backward compatible**: Can disable constant memory, shared memory still works
- **Low overhead**: Kernel upload is <100 bytes, negligible transfer time

### Original Strategy (for reference)
- Gaussian kernels generated in `_computeKernels()` (convolve.c)
- Passed as pointers to CUDA kernels (global memory)
- Kernel size: typically 15-21 floats (60-84 bytes)
- Read by all threads in convolution

### Implementation Strategy
1. **Declare constant memory**:
   ```cuda
   // In convolve_gpu.cu at file scope
   __constant__ float d_gaussKernel[64];      // Max kernel size
   __constant__ float d_gaussDerivKernel[64]; // For gradients
   __constant__ int d_kernelWidth;
   ```

2. **Copy kernel to constant memory**:
   ```cuda
   // Host function to initialize (call once per tracking context)
   extern "C" void cudaSetGaussianKernel(const float* h_kernel, int width) {
       cudaMemcpyToSymbol(d_gaussKernel, h_kernel, width * sizeof(float));
       cudaMemcpyToSymbol(d_kernelWidth, &width, sizeof(int));
   }
   ```

3. **Update kernel signatures**:
   ```cuda
   // Remove kernel pointer parameters
   __global__ void cudaConvolveHorizShared(
       const float* input, float* output,
       int width, int height
       // No kernel parameter - use d_gaussKernel
   ) {
       // Access via d_gaussKernel[k] instead of kernel[k]
       for (int k = 0; k < d_kernelWidth; k++) {
           sum += tile[ty][tx + k] * d_gaussKernel[k];
       }
   }
   ```

4. **Initialize in tracking context**:
   - Call `cudaSetGaussianKernel()` in `KLTCreateContext()` or before first pyramid build
   - Separate calls for Gaussian and derivative kernels

5. **Expected impact**: 10-20% speedup on convolution (reduces latency vs global memory)

### Verification
- Check constant cache hit rate with ncu: `ncu --metrics l2_utilization`
- Verify kernel values match host values
- Test with different sigma values

---

## Optimization 4: Coalesced Memory Access

### Target Files
- `convolve_gpu.cu` (vertical convolution)
- `interpolate_cuda.cu` (if not using texture memory)
- `mineigenvalue_cuda.cu`

### Current State
- Vertical convolution: threads access `input[row * width + col]` with stride = width (non-coalesced)
- Naive interpolation: random access patterns for feature coordinates
- Poor memory bandwidth utilization (~30-40% in profile)

### Implementation Strategy
1. **For vertical convolution**:
   - **Transpose input approach**:
     ```cuda
     // Option A: Transpose before vertical pass
     __global__ void transposeKernel(const float* input, float* output, 
                                      int width, int height) {
         __shared__ float tile[TILE_DIM][TILE_DIM+1]; // +1 to avoid bank conflicts
         int x = blockIdx.x * TILE_DIM + threadIdx.x;
         int y = blockIdx.y * TILE_DIM + threadIdx.y;
         
         if (x < width && y < height)
             tile[threadIdx.y][threadIdx.x] = input[y * width + x];
         __syncthreads();
         
         x = blockIdx.y * TILE_DIM + threadIdx.x;
         y = blockIdx.x * TILE_DIM + threadIdx.y;
         if (x < height && y < width)
             output[y * height + x] = tile[threadIdx.x][threadIdx.y];
     }
     // Then vertical becomes horizontal on transposed data
     ```
   
   - **Or optimize access pattern**:
     ```cuda
     // Option B: Load coalesced, compute vertical
     // Map threads to columns (not rows)
     int col = blockIdx.x * blockDim.x + threadIdx.x;
     int row_start = blockIdx.y * TILE_HEIGHT;
     
     // Load column into shared memory (coalesced per warp)
     if (col < width) {
         for (int i = 0; i < TILE_HEIGHT; i++) {
             int row = row_start + i;
             if (row < height)
                 s_column[i] = input[row * width + col];
         }
     }
     ```

2. **For interpolation (if not using texture)**:
   - Sort features by y-coordinate to improve locality
   - Process in batches with similar coordinates
   - Use shared memory to cache image regions

3. **General pattern**:
   - Threads in same warp should access consecutive memory addresses
   - Use `(blockIdx.x * blockDim.x + threadIdx.x)` for linear indexing
   - Avoid stride access; prefer row-major sequential

4. **Expected impact**: 20-50% speedup on vertical convolution

### Verification
- Check global memory load efficiency: `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`
- Should be >80% for coalesced access
- Compare bandwidth utilization before/after

---

## Optimization 5: Persistent GPU Buffers

### Target Files
- `interpolate_cuda.cu`
- `convolve_gpu.cu`
- `mineigenvalue_cuda.cu`
- `klt.c` (tracking context)
- `trackFeatures.c`

### Current State
- Likely allocates/frees buffers per kernel call (e.g., cudaMalloc in each function)
- Overhead from repeated allocation (~5-10% of execution time)
- No buffer reuse across frames

### Implementation Strategy
1. **Add GPU buffer fields to KLT_TrackingContext**:
   ```c
   // In klt.h
   typedef struct {
       // Existing fields...
       
       // GPU persistent buffers
       void* d_img_data;          // Device image buffer
       void* d_gradx_data;        // Device gradient X
       void* d_grady_data;        // Device gradient Y
       void* d_tmp_buffer;        // Temp for convolution
       void* d_feature_coords;    // Feature coordinates
       void* d_results;           // Interpolation results
       size_t d_buffer_size;      // Current allocation size
       cudaTextureObject_t d_texObj; // Texture object for interpolation
   } KLT_TrackingContextRec, *KLT_TrackingContext;
   ```

2. **Allocate buffers once in KLTCreateContext()**:
   ```c
   // In klt.c
   KLT_TrackingContext KLTCreateContext() {
       KLT_TrackingContext tc = (KLT_TrackingContext)malloc(...);
       // Set parameters...
       
       // Allocate GPU buffers (size based on max image dimensions)
       size_t img_size = MAX_WIDTH * MAX_HEIGHT * sizeof(float);
       cudaMalloc(&tc->d_img_data, img_size);
       cudaMalloc(&tc->d_gradx_data, img_size);
       cudaMalloc(&tc->d_grady_data, img_size);
       cudaMalloc(&tc->d_tmp_buffer, img_size);
       cudaMalloc(&tc->d_feature_coords, MAX_FEATURES * 2 * sizeof(float));
       cudaMalloc(&tc->d_results, MAX_FEATURES * sizeof(float));
       tc->d_buffer_size = img_size;
       
       // Initialize texture object (if using texture memory)
       cudaResourceDesc resDesc;
       memset(&resDesc, 0, sizeof(resDesc));
       resDesc.resType = cudaResourceTypePitch2D;
       // Configure and create texture...
       
       return tc;
   }
   ```

3. **Free buffers in KLTFreeContext()**:
   ```c
   void KLTFreeTrackingContext(KLT_TrackingContext tc) {
       cudaFree(tc->d_img_data);
       cudaFree(tc->d_gradx_data);
       cudaFree(tc->d_grady_data);
       cudaFree(tc->d_tmp_buffer);
       cudaFree(tc->d_feature_coords);
       cudaFree(tc->d_results);
       if (tc->d_texObj) cudaDestroyTextureObject(tc->d_texObj);
       free(tc);
   }
   ```

4. **Modify CUDA function signatures**:
   ```cuda
   // Pass context instead of allocating internally
   extern "C" void cudaNaiveInterpolate(
       KLT_TrackingContext tc,  // Access tc->d_img_data, tc->d_results
       int width, int height,
       const float* h_coords_x, const float* h_coords_y,
       float* h_results, int numPoints
   ) {
       // Use tc->d_feature_coords instead of cudaMalloc
       cudaMemcpy(tc->d_feature_coords, h_coords_x, ...);
       // Launch kernel with tc->d_img_data
       // Copy results from tc->d_results
   }
   ```

5. **Handle dynamic sizing**:
   ```c
   // Reallocate only if needed
   if (new_size > tc->d_buffer_size) {
       cudaFree(tc->d_img_data);
       cudaMalloc(&tc->d_img_data, new_size);
       tc->d_buffer_size = new_size;
   }
   ```

6. **Expected impact**: 5-10% speedup (reduces allocation overhead)

### Verification
- Profile with nsys: check cudaMalloc/cudaFree frequency
- Should see allocations only at context creation/destruction
- Verify memory usage with `nvidia-smi` (should be constant during tracking)

---

## Optimization 6: Async Kernel Launches and cudaMemCpyAsync

### Target Files
- `trackFeatures.c`
- `pyramid.c`
- All CUDA kernel files

### Current State
- Synchronous kernel launches block CPU
- cudaMemcpy blocks both CPU and GPU
- No overlap of computation and transfers

### Implementation Strategy
1. **Create CUDA streams**:
   ```c
   // In KLT_TrackingContext
   typedef struct {
       // Existing fields...
       cudaStream_t stream_pyramid;   // For pyramid construction
       cudaStream_t stream_tracking;  // For feature tracking
       cudaStream_t stream_transfer;  // For data transfers
   } KLT_TrackingContextRec;
   
   // In KLTCreateContext()
   cudaStreamCreate(&tc->stream_pyramid);
   cudaStreamCreate(&tc->stream_tracking);
   cudaStreamCreate(&tc->stream_transfer);
   ```

2. **Use pinned host memory**:
   ```c
   // Replace malloc with cudaHostAlloc for frequently transferred data
   float* h_img_data;
   cudaHostAlloc(&h_img_data, img_size, cudaHostAllocDefault);
   // Use for image data, feature coordinates
   ```

3. **Launch kernels asynchronously**:
   ```cuda
   // In convolution (pyramid.c)
   cudaConvolveHorizShared<<<grid, block, 0, tc->stream_pyramid>>>(
       tc->d_img_data, tc->d_tmp_buffer, width, height
   );
   cudaConvolveVertShared<<<grid, block, 0, tc->stream_pyramid>>>(
       tc->d_tmp_buffer, tc->d_img_data, width, height
   );
   // No cudaDeviceSynchronize() - let stream handle dependencies
   ```

4. **Overlap transfers with computation**:
   ```c
   // Example: Load next image while processing current
   cudaMemcpyAsync(tc->d_next_img, h_next_img, img_size, 
                   cudaMemcpyHostToDevice, tc->stream_transfer);
   
   // Track features on current image (different stream)
   cudaTrackFeatures<<<..., tc->stream_tracking>>>(tc->d_img_data, ...);
   
   // Synchronize only when needed
   cudaStreamSynchronize(tc->stream_tracking); // Wait for tracking
   ```

5. **Pipeline stages**:
   ```c
   // Frame processing pipeline
   // Stage 1: Transfer frame N+1 to GPU (stream_transfer)
   // Stage 2: Build pyramid for frame N (stream_pyramid)
   // Stage 3: Track features on frame N (stream_tracking)
   // Overlap all three for different frames
   ```

6. **Synchronization points**:
   ```c
   // Only sync when CPU needs results
   cudaStreamSynchronize(tc->stream_tracking); // Before reading features back
   // Or use events for fine-grained control
   cudaEventRecord(event, stream);
   cudaEventSynchronize(event);
   ```

7. **Expected impact**: 10-30% speedup (hides transfer latency)

### Verification
- Profile with nsys: check timeline for overlapping kernels/transfers
- Look for gaps in GPU utilization (should be minimal)
- Verify correctness with sequential reference

---

## Optimization 7: Block and Grid Size Optimization [DONE]

### Target Files
- All CUDA kernel files (`interpolate_cuda.cu`, `convolve_gpu.cu`, `mineigenvalue_cuda.cu`)

### Implementation Status: ✅ COMPLETE
**Date Completed**: December 2024  
**Documentation**: See `BLOCK_GRID_OPTIMIZATION.md` for comprehensive details

### What Was Implemented
1. **Interpolation kernels** (`interpolate_cuda.cu`):
   - ✅ Adaptive block sizing: 128 threads for <100 features, 256 for ≥100
   - ✅ Applied to all 4 variants (naive, texture, persistent, texture+persistent)
   - **Expected Impact**: Reduced resource waste on small workloads

2. **Horizontal convolution** (`convolve_gpu.cu`):
   - ✅ Aspect-aware block sizing:
     - 32×8 blocks for wide images (ncols ≥ nrows×2) → Better horizontal coalescing
     - 16×16 blocks for balanced images
     - 16×16 blocks for small images (<512×512)
   - **Expected Impact**: 1.15-1.25× speedup on wide images

3. **Vertical convolution** (`convolve_gpu.cu`):
   - ✅ Aspect-aware block sizing:
     - 8×32 blocks for tall images (nrows ≥ ncols×2) → Better vertical locality
     - 16×16 blocks for balanced/wide images
     - 16×16 blocks for small images (<512×512)
   - **Expected Impact**: 1.15-1.30× speedup on tall images

4. **Eigenvalue kernel** (`mineigenvalue_cuda.cu`):
   - ✅ Optimal 16×16 square blocks (isotropic window access)
   - ✅ Comprehensive documentation of rationale
   - **Expected Impact**: Already optimal, no change

### Performance Impact
- **Per-kernel speedup**: 1.0-1.30× depending on image dimensions
- **Overall speedup**: 1.10-1.20× on typical mixed workloads
- **Cumulative with Opt 1+5+7**: **8-12× vs baseline CPU**

### Current State
- Block/grid optimization is **COMPLETE**
- All kernels use optimal launch configurations
- Adaptive sizing handles varied workloads (small/large images, few/many features)

### Original Strategy (for reference)
1. **For 1D workloads (interpolation, feature processing)**:
   ```cuda
   // Choose block size based on occupancy
   int blockSize = 256; // Start with 256
   int numBlocks = (numPoints + blockSize - 1) / blockSize;
   
   // Or use occupancy API
   int minGridSize, optimalBlockSize;
   cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optimalBlockSize, 
                                       cudaNaiveInterpolate, 0, 0);
   numBlocks = (numPoints + optimalBlockSize - 1) / optimalBlockSize;
   cudaNaiveInterpolate<<<numBlocks, optimalBlockSize>>>(...);
   ```

2. **For 2D workloads (convolution, image processing)**:
   ```cuda
   // Tile-based sizing
   dim3 blockDim(16, 16); // 256 threads, good for shared memory
   dim3 gridDim((width + 15) / 16, (height + 15) / 16);
   
   // Adjust based on image size
   if (width < 512 && height < 512) {
       blockDim = dim3(8, 8); // Smaller tiles for small images
   }
   ```

3. **Occupancy considerations**:
   ```cuda
   // Check with ncu:
   // - Theoretical occupancy (based on registers, shared memory)
   // - Achieved occupancy (actual utilization)
   // Target: >50% for memory-bound, >75% for compute-bound
   
   // Reduce registers if needed
   __launch_bounds__(256, 4) // 256 threads, 4 blocks per SM
   __global__ void myKernel(...) {
       // Compiler optimizes for this config
   }
   ```

4. **Workload-specific tuning**:
   ```cuda
   // For small feature counts (<100)
   if (numFeatures < 100) {
       blockSize = 64; // Avoid underutilization
   }
   
   // For large images (>1024x1024)
   if (width * height > 1024*1024) {
       blockDim = dim3(32, 32); // Larger tiles
   }
   ```

5. **Expected impact**: 10-30% speedup (better occupancy)

### Verification
- Profile with ncu: `ncu --metrics sm_efficiency,achieved_occupancy`
- Target occupancy >50% for memory-bound kernels
- Experiment with block sizes: 64, 128, 256, 512, 1024

---

## Optimization 8: Eigenvalue Kernel Optimization

### Target Files
- `mineigenvalue_cuda.cu`
- `mineigenvalue_cuda.h`
- `selectGoodFeatures.c`

### Current State
- Computes minimum eigenvalue for all pixels (expensive)
- Uses global memory for gradient matrices
- No shared memory or reductions
- Bottleneck: ~5% of execution time

### Implementation Strategy
1. **Shared memory for window aggregation**:
   ```cuda
   __global__ void cudaMinEigenvalueShared(
       const float* gradx, const float* grady,
       float* eigenvalues,
       int width, int height, int window_hw
   ) {
       __shared__ float s_gxx[TILE_SIZE][TILE_SIZE];
       __shared__ float s_gxy[TILE_SIZE][TILE_SIZE];
       __shared__ float s_gyy[TILE_SIZE][TILE_SIZE];
       
       int tx = threadIdx.x, ty = threadIdx.y;
       int col = blockIdx.x * blockDim.x + tx;
       int row = blockIdx.y * blockDim.y + ty;
       
       // Load gradients into shared memory
       if (col < width && row < height) {
           int idx = row * width + col;
           float gx = gradx[idx];
           float gy = grady[idx];
           s_gxx[ty][tx] = gx * gx;
           s_gxy[ty][tx] = gx * gy;
           s_gyy[ty][tx] = gy * gy;
       }
       __syncthreads();
   ```

2. **Compute structure tensor with shared memory**:
   ```cuda
       // Compute window sum (structure tensor)
       if (col < width && row < height) {
           float sum_gxx = 0, sum_gxy = 0, sum_gyy = 0;
           
           // Sum over window (use shared memory)
           for (int dy = -window_hw; dy <= window_hw; dy++) {
               for (int dx = -window_hw; dx <= window_hw; dx++) {
                   int sy = ty + dy;
                   int sx = tx + dx;
                   if (sy >= 0 && sy < blockDim.y && 
                       sx >= 0 && sx < blockDim.x) {
                       sum_gxx += s_gxx[sy][sx];
                       sum_gxy += s_gxy[sy][sx];
                       sum_gyy += s_gyy[sy][sx];
                   }
               }
           }
   ```

3. **Compute eigenvalue analytically**:
   ```cuda
           // Min eigenvalue of 2x2 matrix: λ_min = (trace - sqrt(trace² - 4*det)) / 2
           float trace = sum_gxx + sum_gyy;
           float det = sum_gxx * sum_gyy - sum_gxy * sum_gxy;
           float discriminant = trace * trace - 4.0f * det;
           float min_eigenval = (trace - sqrtf(fmaxf(discriminant, 0.0f))) * 0.5f;
           
           eigenvalues[row * width + col] = min_eigenval;
       }
   }
   ```

4. **Parallel sorting/selection**:
   ```cuda
   // For selecting top N features
   // Option A: Use thrust::sort on eigenvalues
   #include <thrust/device_vector.h>
   #include <thrust/sort.h>
   thrust::device_vector<float> d_eigen(eigenvalues, eigenvalues + width*height);
   thrust::sort(d_eigen.begin(), d_eigen.end(), thrust::greater<float>());
   
   // Option B: Use parallel reduction for top-k
   // Implement partial sort with shared memory reductions
   ```

5. **Enforce minimum distance**:
   ```cuda
   // After sorting, mark nearby features
   __global__ void enforceMinDistance(
       float* x_coords, float* y_coords,
       int* valid_flags,
       int numFeatures, float mindist
   ) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx >= numFeatures || !valid_flags[idx]) return;
       
       // Check all higher-ranked features
       for (int i = 0; i < idx; i++) {
           if (valid_flags[i]) {
               float dx = x_coords[idx] - x_coords[i];
               float dy = y_coords[idx] - y_coords[i];
               if (dx*dx + dy*dy < mindist*mindist) {
                   valid_flags[idx] = 0; // Mark as invalid
                   return;
               }
           }
       }
   }
   ```

6. **Expected impact**: 2-3x speedup on feature selection (5% → 1.5-2.5%)

### Verification
- Compare eigenvalues with CPU version (should match within epsilon)
- Verify selected features are correctly spaced (mindist constraint)
- Check shared memory usage with ncu

---

## Optimization 9: Additional Memory Transfer/Communication Optimization

### Target Files
- `trackFeatures.c`
- `pyramid.c`
- All CUDA kernel files

### Current State
- Synchronous transfers block execution
- Repeated small transfers (inefficient)
- No batching of data

### Implementation Strategy
1. **Batch transfers**:
   ```c
   // Instead of transferring each image separately
   // BAD:
   cudaMemcpy(d_img0, h_img0, img_size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_img1, h_img1, img_size, cudaMemcpyHostToDevice);
   
   // GOOD:
   // Copy entire pyramid at once
   size_t pyramid_size = 0;
   for (int level = 0; level < nLevels; level++) {
       pyramid_size += pyramid[level]->ncols * pyramid[level]->nrows * sizeof(float);
   }
   cudaMemcpy(d_pyramid_data, h_pyramid_data, pyramid_size, cudaMemcpyHostToDevice);
   ```

2. **Keep data on GPU in sequential mode**:
   ```c
   // In KLT sequential mode, cache pyramid on GPU
   if (tc->sequentialMode) {
       // Don't transfer pyramid_last back to host
       // Keep on GPU: tc->d_pyramid_last
       // Reuse directly in next frame
   }
   ```

3. **Minimize round-trips**:
   ```c
   // BAD: Transfer features back and forth
   cudaMemcpy(h_features, d_features, size, cudaMemcpyDeviceToHost);
   // Process on CPU
   cudaMemcpy(d_features, h_features, size, cudaMemcpyHostToDevice);
   
   // GOOD: Keep on GPU, process with kernel
   cudaProcessFeatures<<<...>>>(d_features, ...);
   ```

4. **Use pinned memory for frequent transfers**:
   ```c
   // Allocate pinned memory for image data
   float* h_img_pinned;
   cudaHostAlloc(&h_img_pinned, img_size, cudaHostAllocDefault);
   // Up to 2x faster transfers than pageable memory
   ```

5. **Zero-copy memory (for small data)**:
   ```c
   // For small, infrequently accessed data
   float* h_params;
   cudaHostAlloc(&h_params, sizeof(params), cudaHostAllocMapped);
   // Access directly from kernel (no explicit transfer)
   cudaHostGetDevicePointer(&d_params, h_params, 0);
   ```

6. **Profile-guided optimization**:
   ```bash
   # Identify transfer bottlenecks with nsys
   nsys profile --trace=cuda,nvtx ./example3
   # Look for:
   # - Large gaps between kernels (indicates transfers)
   # - cudaMemcpy taking >10% of time
   ```

7. **Expected impact**: 5-15% speedup (reduced transfer overhead)

### Verification
- Profile with nsys: check transfer time percentage
- Should be <5% of total execution time
- Verify GPU memory usage is reasonable (not exceeding limits)

---

## Implementation Order and Testing

### Phase 1: Foundation (Week 1)
1. Persistent GPU buffers (Opt 5)
2. Constant memory for kernels (Opt 3)
3. Block/grid size tuning (Opt 7)
- **Test**: Verify correctness, profile baseline

### Phase 2: High-Impact Optimizations (Week 2)
4. Texture memory for interpolation (Opt 1)
5. Shared memory tiling for convolution (Opt 2)
6. Coalesced memory access (Opt 4)
- **Test**: Compare speedup vs naive, check correctness

### Phase 3: Advanced Optimizations (Week 3)
7. Async launches and streams (Opt 6)
8. Memory transfer optimization (Opt 9)
9. Eigenvalue kernel optimization (Opt 8)
- **Test**: Profile with ncu/nsys, measure overall speedup

### Phase 4: Tuning and Validation
- Fine-tune block sizes per kernel
- Optimize stream scheduling
- Validate on different GPUs (T4, V100, A100)
- Compare with CPU baseline: target 50-100x speedup

---

## Profiling Commands

```bash
# Full workflow on Colab
make clean && make example3

# Baseline timing
time ./example3

# Timeline profiling (nsys)
make nsys-example3
nsys stats --report cuda_gpu_kern_sum profile-example3-nsys.nsys-rep

# Kernel-level profiling (ncu)
make ncu-quick-example3
# View in GUI: ncu-ui profile-example3-ncu.ncu-rep

# Specific metrics
ncu --metrics achieved_occupancy,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
    ./example3

# Memory bandwidth
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./example3
```

---

## Expected Overall Speedup

- **Naive CUDA (current)**: 5-10x vs CPU
- **After texture memory**: 10-20x
- **After shared memory tiling**: 20-40x
- **After all optimizations**: 50-150x (memory-bound limit)

Target: <100ms for 150 features, 10 frames on Tesla T4 (vs ~5-10s CPU baseline)

---

## Common Pitfalls

1. **Shared memory bank conflicts**: Use `+1` padding in shared arrays
2. **Race conditions**: Always `__syncthreads()` after shared memory writes
3. **Occupancy too low**: Reduce register usage with `__launch_bounds__`
4. **Uncoalesced access**: Verify with ncu memory metrics
5. **Over-synchronization**: Use streams instead of `cudaDeviceSynchronize()`
6. **Memory leaks**: Always free persistent buffers in context destructor
7. **Incorrect boundary handling**: Test with small images (edge cases)
8. **Floating point precision**: Compare with epsilon tolerance, not exact equality

---

## Validation Checklist

- [ ] Output images match CPU version (visual inspection)
- [ ] Feature coordinates match within 0.01 pixels
- [ ] Feature status codes match (TRACKED, NOT_FOUND, etc.)
- [ ] No memory leaks (valgrind or cuda-memcheck)
- [ ] Occupancy >50% for all kernels (ncu)
- [ ] Memory bandwidth >60% of peak (ncu)
- [ ] Overall speedup >50x vs CPU baseline
- [ ] Works on T4, V100, A100 GPUs
- [ ] No errors with cuda-memcheck: `cuda-memcheck ./example3`
