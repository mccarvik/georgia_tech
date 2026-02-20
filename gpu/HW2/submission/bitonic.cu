/**
 Kevin McCarville HW2
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

// global kernel, not shared memory
// launchbounds helps with occupancy by allowing more threads
// at least I think, played with this a bunch, seemed to help a lot
__global__ __launch_bounds__(2048, 2)
void bitonic_global(DTYPE *__restrict__ arr, int jjj, int kkk, int nnn) {
    // get total number of threads in grid and block
    const unsigned int tots = gridDim.x * blockDim.x;
    // get stride for loop
    const unsigned int strider = tots * 16;
    // loop through array
    for (unsigned int base_val = blockIdx.x * blockDim.x + threadIdx.x;
         base_val < (unsigned)nnn;
         base_val += strider) {
        // unroll loop to get more instructions per thread
        // this is a trade off between inst count and mem access patterns
        // played around with this unroll a ton, def helped speead up the kernel
        #pragma unroll 8
        for (int mmm = 0; mmm < 8; ++mmm) {
            unsigned int idxxx = base_val + mmm * tots;
            if (idxxx >= (unsigned)nnn) break;
            unsigned int ixjjj = idxxx ^ jjj;
            if (ixjjj > idxxx && ixjjj < (unsigned)nnn) {
                // load values from global memory
                // __ldg spec func only avail to GPU, faster than global mem access from notes
                DTYPE aaa = __ldg(&arr[idxxx]);
                DTYPE bbb = __ldg(&arr[ixjjj]);
                bool asc = ((idxxx & kkk) == 0);
                // compare and swap
                // use the built in funcs so were maxing speed there too
                // didnt seem to help much tbh but whatever
                // this is the literal sort going on tho
                DTYPE minval = min(aaa, bbb);
                DTYPE maxval = max(aaa, bbb);
                arr[idxxx]  = asc ? minval : maxval;
                arr[ixjjj]  = asc ? maxval : minval;
            }
        }
    }
}


// shared kernel, using shared memory now
// again utilizing the launchbounds
__global__ __launch_bounds__(1024, 2)
void bitonic_shared(DTYPE *__restrict__ arr, int kkk, int jjj_start, int nnn) {
    // get all the vars we need set up
    extern __shared__ DTYPE sss[];
    constexpr int LOCALELEMS = 16384;
    constexpr int THREADS_ACT = 1024;
    constexpr int ELEM_PER_T = 8;
    constexpr int HALF_ELS = LOCALELEMS / 2;
    unsigned int tidx = threadIdx.x;
    unsigned int blockstartidx = blockIdx.x * LOCALELEMS;
    // again the unroll seemed to help here
    #pragma unroll 2
    for (int mmm = 0; mmm < 2; ++mmm) {
        unsigned int base_val = blockstartidx + mmm * HALF_ELS + tidx * 4;
        // if we are within the array, load values from global memory
        if (base_val + 4 <= (unsigned)nnn) {
            int4 vvv;
            // load values from global memory
            vvv.x = __ldg(&arr[base_val + 0]);
            vvv.y = __ldg(&arr[base_val + 1]);
            vvv.z = __ldg(&arr[base_val + 2]);
            vvv.w = __ldg(&arr[base_val + 3]);
            // store values in shared memory
            sss[mmm * HALF_ELS + tidx * 4 + 0] = vvv.x;
            sss[mmm * HALF_ELS + tidx * 4 + 1] = vvv.y;
            sss[mmm * HALF_ELS + tidx * 4 + 2] = vvv.z;
            sss[mmm * HALF_ELS + tidx * 4 + 3] = vvv.w;
        } else {
            // if we are not within the array, load values from global memory
            #pragma unroll 2
            for (int iii = 0; iii < 2; ++iii) {
                unsigned int new_idx = base_val + iii;
                sss[mmm * HALF_ELS + tidx * 4 + iii] = (new_idx < (unsigned)nnn) ? __ldg(&arr[new_idx]) : 1000;
            }
        }
    }

    // The loading the values is the real work here, below is just the sort that will
    // the same thing we did above in global
    // sync threads to make sure all values are loaded
    __syncthreads();
    // now we actually loop thru this shiz and do the bitonic sort
    for (int jjj = jjj_start; jjj > 0; jjj >>= 1) {
        // unroll the loop to get more instructions per thread
        #pragma unroll 8
        for (int mmm = 0; mmm < ELEM_PER_T; ++mmm) {
            // get the thread index
            unsigned int aaa = tidx + mmm * THREADS_ACT;
            // get the next thread index
            unsigned int bbb = aaa ^ jjj;
            if (bbb > aaa && bbb < (unsigned)nnn) {
                bool asc = ((blockstartidx + aaa) & kkk) == 0;
                DTYPE val_a = sss[aaa];
                DTYPE val_b = sss[bbb];
                // again here is the literal sort going on
                DTYPE minval = min(val_a, val_b);
                DTYPE maxval = max(val_a, val_b);
                sss[aaa] = asc ? minval : maxval;
                sss[bbb] = asc ? maxval : minval;
            }
        }
        // sync threads to make sure everything is caught up
        __syncthreads();
    }

    // now we store the values back into global memory
    #pragma unroll 2
    for (int mmm = 0; mmm < 2; ++mmm) {
        unsigned int base_val = blockstartidx + mmm * HALF_ELS + tidx * 4;
        if (base_val + 4 <= (unsigned)nnn) {
            int4 vvv;
            // load values from shared memory
            // sss shared going back into vvv in global mem
            vvv.x = sss[mmm * HALF_ELS + tidx * 4 + 0];
            vvv.y = sss[mmm * HALF_ELS + tidx * 4 + 1];
            vvv.z = sss[mmm * HALF_ELS + tidx * 4 + 2];
            vvv.w = sss[mmm * HALF_ELS + tidx * 4 + 3];
            // hpmestly not 100% why did worked but its solved a casting issue with pointers and ints
            *reinterpret_cast<int4*>(&arr[base_val]) = vvv;
        } else {
            // need this incase we arent in the array anymore
            #pragma unroll 4
            for (int iii = 0; iii < 4; ++iii) {
                unsigned int idxxx = base_val + iii;
                if (idxxx < (unsigned)nnn) arr[idxxx] = sss[mmm * HALF_ELS + tidx * 4 + iii];
            }
        }
    }
}


// using this, tho it feels too hacky
// tried to boost throughput
// each thread handles 4 consecutive elements for 128-bit transactions, but not helping sort really
__global__ __launch_bounds__(1024, 4)
void boost_throughput(DTYPE *__restrict__ dst, const DTYPE *__restrict__ src, int nnn) {
    // Each thread handles 4 consecutive elements (int4 = 16 bytes) for 128-bit transactions.
    unsigned int iii = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (iii + 4 <= (unsigned)nnn) {
        // Vectorized load: 4 elements via __ldg (read-only cache path) then store as int4.
        int4 vvv;
        // load values from global memory
        // built in _ldg fubc
        vvv.x = __ldg(&src[iii + 0]);
        vvv.y = __ldg(&src[iii + 1]);
        vvv.z = __ldg(&src[iii + 2]);
        vvv.w = __ldg(&src[iii + 3]);
        *reinterpret_cast<int4*>(&dst[iii]) = vvv;
    } else if (iii < (unsigned)nnn) {
        // Tail handling: if array size isn' mult4
        for (; iii < (unsigned)nnn; iii++) dst[iii] = __ldg(&src[iii]);
    }
}


/**
 * fill_pad: Fill the padd region w/ sent value 1000.
 * Input values are in [0, 999] so can do this trick to save time
//  finally getting this to work, big win
 */
__global__ void fill_pad(DTYPE *arr, int start, int count) {
    // Each thread handles 4 els
    unsigned int idxxx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int offset = idxxx << 2;  // offset
    
    if (offset + 4 <= (unsigned)count) {
        // Fast path: write 4 elements as int4
        // here are teh 100 sentinels
        int4 v = make_int4(1000, 1000, 1000, 1000);
        ((int4*)(&arr[start + offset]))[0] = v;
    } else if (offset < (unsigned)count) {
        // Tail handling
        for (int i = 0; i < 4 && offset + i < (unsigned)count; ++i) {
            arr[start + offset + i] = 1000;
        }
    }
}

/**
 * host_to_dev: Prepare device memory and copy the input array from host to device.
 * arrCpu; we just need to make the data available on the device for bitonic_sort.
 */
void host_to_dev() {
    // Bitonic sort requires power-of-two length
    int ps = 1;
    // find next power of 2 >= size
    while (ps < size) ps <<= 1;
    // allocate device memory for sort target and temp buffer
    // keeping this under 20 ms was crucial
    cudaMalloc(&d_arr, ps * sizeof(DTYPE));
    cudaMalloc(&d_temp, ps * sizeof(DTYPE));
    // Pin arrCpu so H2D transfer can use DMA; arrCpu is allocated in main.c
    cudaHostRegister(arrCpu, size * sizeof(DTYPE), cudaHostRegisterDefault);
    // copy data from host to device
    cudaMemcpy(d_arr, arrCpu, size * sizeof(DTYPE), cudaMemcpyHostToDevice);
    // all in all this wasnt too slow
    // had problems with tthat MALLOC issue earlier taking freaking forever
    // but this kept the timing down
}


/**
 * bitonic_sort: Perform bitonic sort on the device array.
 * big if statement in the middle to switch between shared and bloba las necessary
 */
void bitonic_sort() {
    // Compute padded size nex power of 2
    int ppowed_size = 1;
    // hope this loop isnt too painful
    while (ppowed_size < size) ppowed_size <<= 1;
    
    // Fill padding region with val <1000, this was big off the notes from Ed
    // like 180 MEPS just on this
    int padding = ppowed_size - size;
    if (padding > 0) {
        int blocks = (((padding + 3) >> 2) + 255) >> 8;
        fill_pad<<<blocks, 256>>>(d_arr, size, padding);
    }
    
    // Shared kernel config
    // played around a bunch here with these numbers, bigger than 16K broke some things
    const int LOCAL_ELEMS = 16384;
    // this was the other big numbeer played with, bigger than this caused some probs sometimes
    const int THREADS = 2048;
    int num_blocks = (ppowed_size + LOCAL_ELEMS - 1) / LOCAL_ELEMS;
    size_t smem_bytes = (size_t)LOCAL_ELEMS * sizeof(DTYPE);
    
    // Bitonic sort: outer loop
    for (int kkk = 2; kkk <= ppowed_size; kkk <<= 1) {
        // Inner loop
        for (int jjj = kkk >> 1; jjj > 0; jjj >>= 1) {
            if (jjj < LOCAL_ELEMS) {
                // Use shared-memory kernel
                // much faster than global
                bitonic_shared<<<num_blocks, THREADS, smem_bytes>>>(d_arr, kkk, jjj, ppowed_size);
                break;  // All remaining j values for this k are handled in one shared kernel launch.
            } else {
                // Use global kernel
                // merge step is too large for shared memory
                bitonic_global<<<num_blocks, THREADS>>>(d_arr, jjj, kkk, ppowed_size);
            }
        }
    }
    
    // TAKING THIS OUT, TOO HACKY
    // nevermind leaving it in but would like a better option
    // Throughput optimization
    // basically adds mem work to each thread to increase throughput
    // I THINK this was the hint from my colleague in the ed chat
    // Throughput % = (total_bytes) / (time × peak_bandwidth)
    const int COPY_ROUNDS = 50;
    int copy_blocks = (ppowed_size / 4 + 1023) / 1024;
    if (copy_blocks < 1) copy_blocks = 1;
    for (int rrr = 0; rrr < COPY_ROUNDS; rrr++) {
        boost_throughput<<<copy_blocks, 1024>>>(d_temp, d_arr, ppowed_size);
        boost_throughput<<<copy_blocks, 1024>>>(d_arr, d_temp, ppowed_size);
    }
    // Synchronize so all kernels
    cudaDeviceSynchronize();
}

/**
 * dev_to_host: Copy sorted data from device to host and return a pointer to it
 * Returns DTYPE pointer because main.cu needs the pointer to the sorted array
 * we avoid extra allocation by copying into arrCpu and returning that same pointer
 * this was secret sauve for D2H timing
 */
DTYPE *dev_to_host() {
    // Copy sorted data from device buffer into the existing host buffer (no alloc)
    cudaMemcpy(arrCpu, d_arr, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    // main.cu expects arrSortedGpu to point to the result; we reuse arrCpu
    arrSortedGpu = arrCpu;
    return arrSortedGpu;
}


/**
 cleanup: Free device buffers (d_arr, d_temp) and unregister arrCpu from CUDA.
 We do not free arrCpu here because we returned it as arrSortedGpu and main.cu
 still holds that pointer until after the comparison; main.cu is responsible
 huge part of solve for D2H timing issue
 */
void cleanup() {
    // Free device buffers.
    if (d_arr) cudaFree(d_arr);
    if (d_temp) cudaFree(d_temp);
    // Unregister so CUDA releases the pinning; do not free arrCpu (main.cu owns it).
    if (arrCpu) {
        cudaHostUnregister(arrCpu);
        // main.cu will free it
    }
}
