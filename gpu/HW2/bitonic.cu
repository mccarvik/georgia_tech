/**
 * 
 * The student is required to add content to this file.  This file is
 * your implementation of the project and will be submitted for grading.
 * 
 */

#include "main.h"
#include "student.h"

/**********************************************************************************
 * 
 * Implement your GPU device kernel(s) here (e.g., the bitonic sort kernel).
 * 
 **********************************************************************************/

// damn right I will
// set up the devive pointer to our array
DTYPE *d_arr;

/*
bitunic sort kernel, set global memeroy
use this for larger stages that dont fit in shared mem
optimized for coalescing n reduce diverg
*/

__global__ void bitonic_sort_global(DTYPE *arr, int jjj, int kkk, int size) {
    // calc global thread index
    // CUDA built ins here
    unsigned int idxxx = blockIdx.x * blockDim.x + threadIdx.x;
    // if thread out of bounds, bounce out
    if (idxxx >= size){  return; }

    // use the XOR
    unsigned int idxjjj = idxxx ^ jjj;
    // only process if idx < idxjjj to avoid redundant comps
    // and make sure were in bounds
    if (idxjjj > idxxx && idxjjj < size) {
        // which direction we going, kkk tells us
        // if (idxxx & kkk) == 0, sort asc; otherwise desc
        bool ascend = ((idxxx & kkk) == 0);

        // now grab vals from global mem
        DTYPE aaa = arr[idxxx];
        DTYPE bbb = arr[idxjjj];

        // compare and swap
        if (ascend ? (aaa > bbb) : (aaa < bbb)) {
            arr[idxxx] = bbb;
            arr[idxjjj] = aaa;
        }
        // otherwise no need
        // this is the sort
    }
}

/*
shared mem bitonic sort kernel
Not sure I need this? idk maybe I will use it later
Kept here for ref
// not using this as of now
// nevermind we might use if if triggers it, idk
*/
__global__ void bitonic_sort_shared(DTYPE *arr, int stage, int size) {
    // setup orig vars
    // thread id and global id
    extern __shared__ DTYPE shared[];
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // grab data abd oyut it in sharddd mem
    shared[tid] = (gid < size) ? arr[gid] : INT_MAX;
    // sync
    __syncthreads();

    // now we sort
    // jjj iterates
    for (int jjj = stage; jjj > 0; jjj >>= 1) {
        // create index jjj
        unsigned int idxjjj = tid ^ jjj;

        // only process val pairs in the block
        if (idxjjj > tid && idxjjj < blockDim.x) {
            // now setup global idx
            unsigned int globtempidx = blockIdx.x * blockDim.x + tid;
            // which direction we going, stage tells us
            bool ascend = ((globtempidx & (1 << (stage + 1))) == 0);
            // from shared mem
            DTYPE aaa = shared[tid];
            DTYPE bbb = shared[idxjjj];

            // compare and swap
            // CnP'd above
            if ((ascend && aaa > bbb) || (!ascend && aaa < bbb)) {
                shared[tid] = bbb;
                shared[idxjjj] = aaa;
            }
            // this is the sort in action
        }
        // sync
        __syncthreads();


    // write back to global mem
    if (gid < size) {
        arr[gid] = shared[tid];
    }
    // this should do it
}


// merg sort function?
// do we nned it? might come back to this


/**********************************************************************************
 * 
 * Implement your utility functions here
 * 
 **********************************************************************************/

// well see what we need here
// nothing yet

int nexpow2(int nnn) {
    // i mean just inc ase well have this
    int pow =1;
    while (pow < nnn) { pow *= 2; }
    return pow;
}

int log2_int(int nnn) {
    // same as above, maybe need this but can prob do it in line
    int log = 0;
    while (nnn > 1) { nnn >>= 1; log++; }
    return log;
}

/**********************************************************************************
 * 
 * Implement the three main program functions
 * 
 **********************************************************************************/



/**
 * This function transfers data from Host to Device
 */
void host_to_dev()
{
    // calc next pow 2, well do it inline
    int padsiz = 1;
    // while less than size, shift left
    while (padsiz < size) { padsiz <<= 1; }

    // alloc mem
    cudaMalloc((void**)&d_arr, padsiz * sizeof(DTYPE));
    // copy data
    // some of these are built ins, not sure they are right
    cudaMemcpy(d_arr, arrCpu, size * sizeof(DTYPE), cudaMemcpyHostToDevice);

    int paddiff = padsiz - size;

    // if padsiz is greater than size, we need to fill the padding with INT_MAX to sort to end
    if (padsiz > size) {
        // alloc padding
        DTYPE *hpadd = (DTYPE*)malloc(paddiff * sizeof(DTYPE));
        // really unsure here but we will see, a lot of default stuff coming from CUDA
        for (int i = 0; i < paddiff; i++) {
            hpadd[i] = INT_MAX;
        }
        // alot of these vars are built ins, not sure they are right
        cudaMemcpy(d_arr + size, hpadd, paddiff * sizeof(DTYPE), cudaMemcpyHostToDevice);
        free(hpadd);
    }

}

/**
 * This function performs the bitonic sort and merge by calling the
 * kernels you have defined in the section above
 */
void bitonic_sort()
{
    // same as above
    int paddsiz = 1
    while (paddsiz < size) { paddsiz <<= 1; }

    // block size of H100 GPU is 512
    int blocksiz = 512;
    int numblcks = (paddsiz + blocksiz - 1) / blocksiz;

    // now we sort
    // outer loop: stage of algs
    // kkk = size of seq
    for (int kkk = 2; kkk <= paddsiz; kkk <<= 1) {
        // inner loop: bitonic merge ops
        // for each stage kkk, we need to do bitonic splits for jjj = kkk/2, kkk/4, yadda yadda yadda
        for (int jjj = kkk >> 1; jjj > 0; jjj >>= 1) {
            // if jjj is small enough, use shared mem for remaining iters
            if (jjj < blocksiz) {
                int sharedmemsize = blocksiz * sizeof(DTYPE);
                // using shared mem kernel
                // really not sure about this part
                bitonic_sort_shared<<<numblcks, blocksiz, sharedmemsize>>>(d_arr, kkk, jjj, paddsiz);
                break; 
                // all remaining jjj iters are handled within the kernel so we can break
            }
            else {
                // use global mem
                bitonic_sort_global<<<numblcks, blocksiz>>>(d_arr, jjj, kkk, paddsiz);
            }
        }
    }
    // ensure all kernels complete before returning this bad boy
    cudaDeviceSynchronize();
}

/**
 * This functiuon transfers the sorted data from Device to Host
 */
DTYPE *dev_to_host()
{
    // should be simpler function
    // alloc mem for sorted array
    arrSortedGpu = (DTYPE*)malloc(size * sizeof(DTYPE));
    // and then move from device to host
    // again a lot of this is straight from CUDA docs
    cudaMemcpy(arrSortedGpu, d_arr, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    // and then return the pointer to the sorted array
    return arrSortedGpu;
}

/**
 * This function frees memory and anything else the student requires 
 * before exiting the program
 */
void cleanup(){
    // free device memory
    cudaFree(d_arr);
    // You may modify/remove these as needed to make your implementation work
    // properly. The defaults provided here allow the skeleton code to compile.    
    // straight forward here
    free(arrCpu);
    free(arrSortedGpu);
}
