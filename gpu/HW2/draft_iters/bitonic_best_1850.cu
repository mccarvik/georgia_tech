/**
 * Name: Kevin McCarville
 * Class: GPU Computing  
 * Assignment: HW2 - Bitonic Sort
 * 
 * NEW: 8192-element shared tiles (8 elems/thread) + heavy ILP/unrolling
 * This is the biggest remaining win for kernel time on H100.
 */

#include "main.h"
#include "student.h"
#include <cstring>

static inline unsigned next_pow2(unsigned u) {
    if (u <= 1u) return 1u;
#ifdef __GNUC__
    return 1u << (32 - __builtin_clz(u - 1u));
#else
    unsigned r = 1u; while (r < u) r <<= 1; return r;
#endif
}

DTYPE *d_arr = nullptr;

/* ====================== GLOBAL KERNEL (high ILP) ====================== */
__global__ __launch_bounds__(1024, 2)
void bitonic_global(DTYPE *__restrict__ arr, int j, int k, int n) {
    const unsigned int total = gridDim.x * blockDim.x;
    const unsigned int stride = total * 4;               // unroll factor 4

    for (unsigned int base = blockIdx.x * blockDim.x + threadIdx.x;
         base < (unsigned)n;
         base += stride) {

        #pragma unroll 4
        for (int m = 0; m < 4; ++m) {
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

/* ====================== SHARED KERNEL (8192 elems/block) ====================== */
__global__ __launch_bounds__(1024, 2)
void bitonic_shared(DTYPE *__restrict__ arr, int k, int j_start, int n) {
    extern __shared__ DTYPE s[];

    constexpr int LOCAL = 8192;
    constexpr int THREADS = 1024;
    constexpr int ELEM_PER_T = 8;      // 8192 / 1024

    unsigned int tid = threadIdx.x;
    unsigned int block_start = blockIdx.x * LOCAL;

    /* Load 8 elements per thread (coalesced) */
    #pragma unroll 8
    for (int m = 0; m < ELEM_PER_T; ++m) {
        unsigned int idx = block_start + tid + m * THREADS;
        s[tid + m * THREADS] = (idx < (unsigned)n) ? __ldg(&arr[idx]) : 1000;
    }
    __syncthreads();

    /* Remaining merges in shared memory */
    for (int j = j_start; j > 0; j >>= 1) {
        /* 8 pairs per thread (covers all 8192 elements) */
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

    /* Write back */
    #pragma unroll 8
    for (int m = 0; m < ELEM_PER_T; ++m) {
        unsigned int idx = block_start + tid + m * THREADS;
        if (idx < (unsigned)n) arr[idx] = s[tid + m * THREADS];
    }
}

/* ====================== PADDING (unchanged) ====================== */
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

/* ====================== HOST <-> DEVICE (UNTOUCHED) ====================== */
void host_to_dev() {
    unsigned padded_size = next_pow2((unsigned)size);
    cudaMalloc(&d_arr, (size_t)padded_size * sizeof(DTYPE));
    cudaHostRegister(arrCpu, size * sizeof(DTYPE), cudaHostRegisterDefault);
    cudaMemcpy(d_arr, arrCpu, size * sizeof(DTYPE), cudaMemcpyHostToDevice);
}

void bitonic_sort() {
    int padded_size = next_pow2((unsigned)size);

    /* padding fill */
    int padding = padded_size - size;
    if (padding > 0) {
        int blocks = (((padding + 3) >> 2) + 255) >> 8;
        fill_pad<<<blocks, 256>>>(d_arr, size, padding);
    }

    /* 8192-element tiles */
    constexpr int LOCAL_ELEMS = 8192;
    constexpr int THREADS = 1024;
    const int num_blocks = (padded_size + LOCAL_ELEMS - 1) / LOCAL_ELEMS;

    for (int k = 2; k <= padded_size; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            if (j < LOCAL_ELEMS) {
                size_t smem = (size_t)LOCAL_ELEMS * sizeof(DTYPE);
                bitonic_shared<<<num_blocks, THREADS, smem>>>(d_arr, k, j, padded_size);
                break;
            } else {
                bitonic_global<<<num_blocks, THREADS>>>(d_arr, j, k, padded_size);
            }
        }
    }
}

DTYPE *dev_to_host() {
    // Use regular malloc + cudaHostRegister (NO cudaMallocHost overhead!)
    arrSortedGpu = (DTYPE*)malloc(size * sizeof(DTYPE));
    cudaHostRegister(arrSortedGpu, size * sizeof(DTYPE), cudaHostRegisterDefault);
    cudaMemcpy(arrSortedGpu, d_arr, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    cudaHostUnregister(arrSortedGpu);
    return arrSortedGpu;
}

void cleanup() {
    if (d_arr) { cudaFree(d_arr); d_arr = nullptr; }
    if (arrSortedGpu) { free(arrSortedGpu); arrSortedGpu = nullptr; }
    if (arrCpu) {
        cudaHostUnregister(arrCpu);
        free(arrCpu);
        arrCpu = nullptr;
    }
}
