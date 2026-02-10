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

}

/**
 * This function performs the bitonic sort and merge by calling the
 * kernels you have defined in the section above
 */
void bitonic_sort()
{

}

/**
 * This functiuon transfers the sorted data from Device to Host
 */
DTYPE *dev_to_host()
{
    // Default value.  You can return any pointer you wish based on
    // your implementation.
    return arrSortedGpu;
}

/**
 * This function frees memory and anything else the student requires 
 * before exiting the program
 */
void cleanup(){
    
    // You may modify/remove these as needed to make your implementation work
    // properly. The defaults provided here allow the skeleton code to compile.    
    free(arrCpu);
    free(arrSortedGpu);
}
