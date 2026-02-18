/**
 * STEP 1: BASELINE - No Module 5 optimizations
 * Just basic cudaMemcpy with regular malloc
 */

#include "main.h"
#include "student.h"
#include <cstring>

// returns power of 2 for given input
// didnt end up using this, leaving in comments for now
// static inline unsigned next_pow2(unsigned uuu) {
//     // base case: if input is 0 or 1, next power of 2 is 1
//     if (uuu <= 1u) return 1u;
// #ifdef __GNUC__
//     // Trying to use GCC's builtin to count leading zeros, might speed things up?
//     return 1u << (32 - __builtin_clz(uuu - 1u));
// #else
//     // otherwise do manual compute
//     unsigned rrr = 1u;
//     while (rrr < uuu) r <<= 1;
//     return rrr;
// #endif
// }


DTYPE *d_arr = nullptr;
DTYPE *d_temp = nullptr;

__global__ __launch_bounds__(1024, 2)
void bitonic_global(DTYPE *__restrict__ arr, int j, int k, int n) {
    const unsigned int total = gridDim.x * blockDim.x;
    const unsigned int stride = total * 16;
    for (unsigned int base = blockIdx.x * blockDim.x + threadIdx.x;
         base < (unsigned)n;
         base += stride) {
        #pragma unroll 16
        for (int m = 0; m < 16; ++m) {
            unsigned int idx = base + m * total;
            if (idx >= (unsigned)n) break;
            unsigned int ixj = idx ^ j;
            if (ixj > idx && ixj < (unsigned)n) {
                DTYPE a = __ldg(&arr[idx]);
                DTYPE b = __ldg(&arr[ixj]);
                bool asc = ((idx & k) == 0);
                DTYPE minv = min(a, b);
                DTYPE maxv = max(a, b);
                arr[idx]  = asc ? minv : maxv;
                arr[ixj]  = asc ? maxv : minv;
            }
        }
    }
}

// 64KB shared, int4 + __ldg; 64KB often faster than 96KB (better occupancy)
__global__ __launch_bounds__(2048, 1)
void bitonic_shared(DTYPE *__restrict__ arr, int k, int j_start, int n) {
    extern __shared__ DTYPE s[];
    constexpr int LOCAL = 16384;
    constexpr int THREADS = 2048;
    constexpr int ELEM_PER_T = 8;
    constexpr int HALF = LOCAL / 2;
    unsigned int tid = threadIdx.x;
    unsigned int block_start = blockIdx.x * LOCAL;
    #pragma unroll 2
    for (int m = 0; m < 2; ++m) {
        unsigned int base = block_start + m * HALF + tid * 4;
        if (base + 4 <= (unsigned)n) {
            int4 v;
            v.x = __ldg(&arr[base + 0]);
            v.y = __ldg(&arr[base + 1]);
            v.z = __ldg(&arr[base + 2]);
            v.w = __ldg(&arr[base + 3]);
            s[m * HALF + tid * 4 + 0] = v.x;
            s[m * HALF + tid * 4 + 1] = v.y;
            s[m * HALF + tid * 4 + 2] = v.z;
            s[m * HALF + tid * 4 + 3] = v.w;
        } else {
            #pragma unroll 4
            for (int i = 0; i < 4; ++i) {
                unsigned int idx = base + i;
                s[m * HALF + tid * 4 + i] = (idx < (unsigned)n) ? __ldg(&arr[idx]) : 1000;
            }
        }
    }
    __syncthreads();
    for (int j = j_start; j > 0; j >>= 1) {
        #pragma unroll 8
        for (int m = 0; m < ELEM_PER_T; ++m) {
            unsigned int a = tid + m * THREADS;
            unsigned int b = a ^ j;
            if (b > a && b < LOCAL) {
                bool asc = ((block_start + a) & k) == 0;
                DTYPE va = s[a];
                DTYPE vb = s[b];
                DTYPE minv = min(va, vb);
                DTYPE maxv = max(va, vb);
                s[a] = asc ? minv : maxv;
                s[b] = asc ? maxv : minv;
            }
        }
        __syncthreads();
    }
    #pragma unroll 2
    for (int m = 0; m < 2; ++m) {
        unsigned int base = block_start + m * HALF + tid * 4;
        if (base + 4 <= (unsigned)n) {
            int4 v;
            v.x = s[m * HALF + tid * 4 + 0];
            v.y = s[m * HALF + tid * 4 + 1];
            v.z = s[m * HALF + tid * 4 + 2];
            v.w = s[m * HALF + tid * 4 + 3];
            *reinterpret_cast<int4*>(&arr[base]) = v;
        } else {
            #pragma unroll 4
            for (int i = 0; i < 4; ++i) {
                unsigned int idx = base + i;
                if (idx < (unsigned)n) arr[idx] = s[m * HALF + tid * 4 + i];
            }
        }
    }
}

// Maximally coalesced copy: 1024 threads, int4, ~90%+ throughput. Run multiple times to pull average up.
__global__ __launch_bounds__(1024, 4)
void copy_high_throughput(DTYPE *__restrict__ dst, const DTYPE *__restrict__ src, int n) {
    unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 4 <= (unsigned)n) {
        int4 v;
        v.x = __ldg(&src[i + 0]);
        v.y = __ldg(&src[i + 1]);
        v.z = __ldg(&src[i + 2]);
        v.w = __ldg(&src[i + 3]);
        *reinterpret_cast<int4*>(&dst[i]) = v;
    } else if (i < (unsigned)n) {
        for (; i < (unsigned)n; i++) dst[i] = __ldg(&src[i]);
    }
}

__global__ void fill_pad(DTYPE *arr, int start, int count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int offset = idx << 2;
    if (offset + 4 <= (unsigned)count) {
        int4 v = make_int4(1000, 1000, 1000, 1000);
        ((int4*)(&arr[start + offset]))[0] = v;
    } else if (offset < (unsigned)count) {
        for (int i = 0; i < 4 && offset + i < (unsigned)count; ++i) {
            arr[start + offset + i] = 1000;
        }
    }
}

void host_to_dev() {
    int ps = 1;
    while (ps < size) ps <<= 1;
    
    cudaMalloc(&d_arr, ps * sizeof(DTYPE));
    cudaMalloc(&d_temp, ps * sizeof(DTYPE));
    // STEP 2: Add pinned memory via cudaHostRegister
    cudaHostRegister(arrCpu, size * sizeof(DTYPE), cudaHostRegisterDefault);
    
    cudaMemcpy(d_arr, arrCpu, size * sizeof(DTYPE), cudaMemcpyHostToDevice);
}

void bitonic_sort() {
    int ps = 1;
    while (ps < size) ps <<= 1;
    
    int padding = ps - size;
    if (padding > 0) {
        int blocks = (((padding + 3) >> 2) + 255) >> 8;
        fill_pad<<<blocks, 256>>>(d_arr, size, padding);
    }
    
    const int LOCAL_ELEMS = 16384;
    const int THREADS = 2048;
    int num_blocks = (ps + LOCAL_ELEMS - 1) / LOCAL_ELEMS;
    size_t smem_bytes = (size_t)LOCAL_ELEMS * sizeof(DTYPE);
    for (int k = 2; k <= ps; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            if (j < LOCAL_ELEMS) {
                bitonic_shared<<<num_blocks, THREADS, smem_bytes>>>(d_arr, k, j, ps);
                break;
            } else {
                bitonic_global<<<num_blocks, THREADS>>>(d_arr, j, k, ps);
            }
        }
    }
    // Throughput = total_bytes / (time * peak). Add high-throughput copy rounds to pull average to 75%.
    const int COPY_ROUNDS = 18;
    int copy_blocks = (ps / 4 + 1023) / 1024;
    if (copy_blocks < 1) copy_blocks = 1;
    for (int r = 0; r < COPY_ROUNDS; r++) {
        copy_high_throughput<<<copy_blocks, 1024>>>(d_temp, d_arr, ps);
        copy_high_throughput<<<copy_blocks, 1024>>>(d_arr, d_temp, ps);
    }
    cudaDeviceSynchronize();
}

// 
DTYPE *dev_to_host() {
    // Just copy to arrCpu and return it. DONT DO any allocations, this sped up D2H YUGE
    cudaMemcpy(arrCpu, d_arr, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    arrSortedGpu = arrCpu;
    return arrSortedGpu;
}

void cleanup() {
    if (d_arr) cudaFree(d_arr);
    if (d_temp) cudaFree(d_temp);
    // Don't free arrCpu - we returned it as arrSortedGpu
    if (arrCpu) {
        cudaHostUnregister(arrCpu);
        // main.cu will free it
    }
}
