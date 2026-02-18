/**
 * This code may not be modified.  You will not be submitting this file.
 */

#include "main.h"
#include "student.h"

// Numbe of elements in the array
int size;

// The array to be sorted by the CPU
DTYPE *arrCpu;

// The array sorted on the GPU
DTYPE *arrSortedGpu;


int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    size = atoi(argv[1]);

    srand(time(NULL));

    // Allocate the array to be sorted
    arrCpu = (DTYPE*)malloc(size * sizeof(DTYPE));

    // arCpu contains the input random array
    for (int i = 0; i < size; i++) {
        arrCpu[i] = rand() % 1000;
    }

    // Timer variables
    float gpuTime, h2dTime, d2hTime, cpuTime = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start);


    // Transfer data (arr_cpu) to device 
    host_to_dev();

    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&h2dTime, start, stop);
    cudaEventRecord(start);
    

    // Perform the bitonic sort
    bitonic_sort();

    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&gpuTime, start, stop);
    cudaEventRecord(start);


    // Transfer sorted data back to host (copied to arrSortedGpu)
     arrSortedGpu = dev_to_host();

    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&d2hTime, start, stop);

    auto startTime = std::chrono::high_resolution_clock::now();
    
    // CPU sort for performance comparison
    std::sort(arrCpu, arrCpu + size);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    cpuTime = cpuTime / 1000;

    int match = 1;
    for (int i = 0; i < size; i++) {
        if (arrSortedGpu[i] != arrCpu[i]) {
            match = 0;
            break;
        }
    }

    // Cleanup - free memory, etc.
    cleanup();


    if (match)
        printf("\033[1;32mFUNCTIONAL SUCCESS\n\033[0m");
    else {
        printf("\033[1;31mFUNCTIONCAL FAIL\n\033[0m");
        return 0;
    }
    
    printf("\033[1;34mArray size         :\033[0m %d\n", size);
    printf("\033[1;34mCPU Sort Time (ms) :\033[0m %f\n", cpuTime);
    float gpuTotalTime = h2dTime + gpuTime + d2hTime;
    int speedup = (gpuTotalTime > cpuTime) ? (gpuTotalTime/cpuTime) : (cpuTime/gpuTotalTime);
    float meps = size / (gpuTotalTime * 0.001) / 1e6;
    printf("\033[1;34mGPU Sort Time (ms) :\033[0m %f\n", gpuTotalTime);
    printf("\033[1;34mGPU Sort Speed     :\033[0m %f million elements per second\n", meps);
    if (gpuTotalTime < cpuTime) {
        printf("\033[1;32mPERF PASSING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;32m %dx \033[1;34mfaster than CPU !!!\033[0m\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
    } else {
        printf("\033[1;31mPERF FAILING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;31m%dx \033[1;34mslower than CPU, optimize further!\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
        return 0;
    }

    return 0;
}
