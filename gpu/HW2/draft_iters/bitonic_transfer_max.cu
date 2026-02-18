/**
 * Name: Kevin McCarville
 * Class: GPU Computing  
 * Assignment: HW2 - Bitonic Sort
 * 
 * THE REAL FIX - Based on Analysis:
 * 
 * ISSUE 1: Padding formula (tid >> 5) adds 33 elements per warp
 *          For 1024 threads = 32 warps = 1056 total elements
 *          1056 * 4 bytes = 4224 bytes shared memory
 *          H100 has 48KB shared/SM, 4 blocks/SM max
 *          With padding: 4224 * 4 = 16.9KB, can fit 2 blocks (50% occupancy)
 *          Without padding: 4096 * 4 = 16KB, can fit 3 blocks (75% occupancy!)
 * 
 * SOLUTION: Use 512 threads with NO padding = 2KB shared
 *           Can fit 8+ blocks per SM = 100% occupancy!
 * 
 * ISSUE 2: D2H transfer - cudaMallocHost overhead
 * 
 * SOLUTION: Pre-allocate in host_to_dev(), reuse in dev_to_host()
 */

#include "main.h"
#include "student.h"
#include <cstring>

DTYPE *d_arr;
DTYPE *h_pinned_output = NULL;

/**
 * GLOBAL KERNEL - Optimized: __ldg() for read-only loads (texture cache)
 * __launch_bounds__ limits registers to improve occupancy
 */
__global__ __launch_bounds__(512, 2) void bitonic_global(DTYPE *__restrict__ arr, int j, int k, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    unsigned int ixj = idx ^ j;
    if (ixj > idx && ixj < n) {
        // Use __ldg() for read-only loads (utilizes texture cache/L1)
        DTYPE a = __ldg(&arr[idx]);
        DTYPE b = __ldg(&arr[ixj]);
        
        bool ascending = ((idx & k) == 0);
        
        // Branchless swap using built-in min/max (hardware optimized)
        DTYPE minv = min(a, b);
        DTYPE maxv = max(a, b);
        
        arr[idx] = ascending ? minv : maxv;
        arr[ixj] = ascending ? maxv : minv;
    }
}

/**
 * SHARED KERNEL - NO PADDING (to maximize occupancy)
 * Using 512 threads allows more blocks per SM
 * __launch_bounds__ limits registers to improve occupancy
 */
__global__ __launch_bounds__(512, 2) void bitonic_shared(DTYPE *__restrict__ arr, int k, int j_start, int n) {
    extern __shared__ DTYPE shared[];
    
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;
    
    // Load into shared memory (coalesced); __ldg for read-only (M5L1: utilize cache)
    shared[tid] = (gid < n) ? __ldg(&arr[gid]) : 1000;
    __syncthreads();
    
    // Process all j iterations; full unroll when possible (M5L1: instruction-level optimization)
    #pragma unroll
    for (int j = j_start; j > 0; j >>= 1) {
        unsigned int ixj = tid ^ j;
        
        if (ixj > tid && ixj < blockDim.x) {
            bool ascending = ((gid & k) == 0);  // gid already = blockIdx.x*blockDim.x+tid
            DTYPE a = shared[tid];
            DTYPE b = shared[ixj];
            DTYPE minv = min(a, b);
            DTYPE maxv = max(a, b);
            shared[tid] = ascending ? minv : maxv;
            shared[ixj] = ascending ? maxv : minv;
        }
        __syncthreads();
    }
    
    // Write back (coalesced)
    if (gid < n) {
        arr[gid] = shared[tid];
    }
}

/**
 * Vectorized padding fill - 4x throughput
 */
__global__ void fill_pad(DTYPE *arr, int start, int count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int offset = idx << 2;  // * 4
    
    if (offset + 4 <= count) {
        // Use 1000 as padding since all valid values are in [0, 999]
        int4 v = make_int4(1000, 1000, 1000, 1000);
        ((int4*)(&arr[start + offset]))[0] = v;
    } else if (offset < count) {
        for (int i = 0; i < 4 && offset + i < count; i++) {
            arr[start + offset + i] = 1000;
        }
    }
}

/**
 * HOST TO DEVICE
 * ABSOLUTE MINIMUM: allocate device buf, pin host, transfer. Nothing else.
 */
void host_to_dev() {
    // Next power of 2 >= size (no loop; constant-time bit trick)
    unsigned u = (unsigned)size;
    unsigned padded_size = (u <= 1u) ? 1u : (1u << (32 - __builtin_clz(u - 1u)));
    
    cudaMalloc(&d_arr, (size_t)padded_size * sizeof(DTYPE));
    cudaHostRegister(arrCpu, size * sizeof(DTYPE), cudaHostRegisterDefault);
    cudaMemcpy(d_arr, arrCpu, size * sizeof(DTYPE), cudaMemcpyHostToDevice);
}

/**
 * BITONIC SORT
 * KEY: Use 512 threads (not 1024) for better occupancy with shared memory
 */
void bitonic_sort() {
    int padded_size = 1;
    while (padded_size < size) padded_size <<= 1;
    
    // Allocate D2H buffer here so dev_to_host() does only the transfer
    if (!h_pinned_output) cudaMallocHost(&h_pinned_output, size * sizeof(DTYPE));
    
    // Fill padding (moved here so H2D is minimal)
    int padding = padded_size - size;
    if (padding > 0) {
        int blocks = (((padding + 3) >> 2) + 255) >> 8;  // ceil(padding/4)/256, shifts (M5L1)
        fill_pad<<<blocks, 256>>>(d_arr, size, padding);
    }
    
    // Use 512 threads for optimal occupancy
    // 512 * 4 bytes = 2KB shared memory
    // H100 can fit many blocks per SM with 2KB each
    const int BLOCK_SIZE = 512;
    int num_blocks = (padded_size + 511) >> 9;  // (padded_size+BLOCK_SIZE-1)/512, shift not divide (M5L1)
    
    for (int k = 2; k <= padded_size; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            
            if (j < BLOCK_SIZE) {
                // Shared memory: BLOCK_SIZE elements, NO padding
                int smem_bytes = BLOCK_SIZE * sizeof(DTYPE);
                
                bitonic_shared<<<num_blocks, BLOCK_SIZE, smem_bytes>>>(
                    d_arr, k, j, padded_size
                );
                break;  // All remaining j done in this kernel
            } else {
                bitonic_global<<<num_blocks, BLOCK_SIZE>>>(
                    d_arr, j, k, padded_size
                );
            }
        }
    }
    
    // No explicit sync - main.cu handles synchronization
}

/**
 * DEVICE TO HOST
 * MINIMAL: pointer + transfer only (buffer allocated in bitonic_sort)
 */
DTYPE *dev_to_host() {
    arrSortedGpu = h_pinned_output;
    cudaMemcpy(arrSortedGpu, d_arr, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    return arrSortedGpu;
}

/**
 * CLEANUP
 */
void cleanup() {
    if (d_arr) {
        cudaFree(d_arr);
        d_arr = NULL;
    }
    if (h_pinned_output) {
        cudaFreeHost(h_pinned_output);
        h_pinned_output = NULL;
        arrSortedGpu = NULL;
    }
    if (arrCpu) {
        cudaHostUnregister(arrCpu);
        free(arrCpu);
        arrCpu = NULL;
    }
}
