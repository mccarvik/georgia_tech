#define TILE_WIDTH 4

// System includes
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <cstring> // Added for strcmp
#include <ctime>
#include <chrono>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Function to compare result matrices from CPU and GPU
bool compareMatrices(float* C, float* D, int size) 
{
    for (int i = 0; i < size; ++i) {
        float error = std::abs(C[i] - D[i]);
        if (error > 1e-3) {
            return false;
        }
    }
    return true;
}

// Function to initialize matrices A and B
void initializeMatrices(float* matrix, int size) 
{
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (int i = 0; i < size; ++i) 
    {
        // Generate a random float number between 0 and 1
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// CPU implementation
void matrixMultiplication(float* A, float* B, float* D, int w) 
{
    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < w; ++j) {
            float sum = 0.0;
            for (int k = 0; k < w; ++k) {
                sum += A[i * w + k] * B[k * w + j];
            }
            D[i * w + j] = sum;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Do not modify code above this line
////////////////////////////////////////////////////////////////////////////////////////////////////////

// GPU implementation
// Implement this kernel function
// A & B are addresses on the host for input matrices, C is the address on the host for output matrix
// matrixWidth is the width of matrices for which matrix multiplication is being performed
__global__ void MatrixMulCUDA(float* C, float* A, float* B, int matrixWidth) 
{
    // Allocate shared memory to be used by a block
    // so we ned to setup our tiles here
    // need to be shared for speed up
    __shared__ float tile_mat_gpu_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_mat_gpu_b[TILE_WIDTH][TILE_WIDTH];
    // Tile width defined at the top as a global

    // blockidx and threadidx are built in variables with CUDA
    int row_calcs_gpu = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col_calcs_gpu = blockIdx.x * TILE_WIDTH + threadIdx.x;
    // this will be our counter for sum
    float sum_ct = 0.0f;

    // matrix width is given, we got the global tile width
    int num_tiles = matrixWidth / TILE_WIDTH;
    for (int ttt = 0; ttt < num_tiles; ttt++) {
        // Load values into the shared memory
        // get the tile inds for x and y
        int tile_idx_x = (ttt * TILE_WIDTH + threadIdx.x);
        int tile_idx_y = (ttt * TILE_WIDTH + threadIdx.y);
        // load the values into the shared memory
        // Perform multiplication and accumulate results into thread-local memory
        tile_mat_gpu_a[threadIdx.y][threadIdx.x] = A[row_calcs_gpu * matrixWidth + tile_idx_x];
        tile_mat_gpu_b[threadIdx.y][threadIdx.x] = B[tile_idx_y * matrixWidth + col_calcs_gpu];

        // Synchronize threads of a block as required
        // I think this is all I need here, will circle back
        __syncthreads();

        // Perform mult and accum into sum
        for (int k_ct = 0; k_ct < TILE_WIDTH; k_ct++) {
            float this_sum = tile_mat_gpu_a[threadIdx.y][k_ct] * tile_mat_gpu_b[k_ct][threadIdx.x];
            sum_ct += this_sum;
        }

        // Needed another one, not sure why
        __syncthreads();
    }

    // stor result in outmat
    C[row_calcs_gpu * matrixWidth + col_calcs_gpu] = sum_ct;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Do not modify code below this line
////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Program main
 */
int main(int argc, char** argv) 
{
    if (argc != 3 || strcmp(argv[1], "-m") != 0) 
    {
        std::cout << "Usage: ./a.out -m <matrix width>" << std::endl;
        return -1;
    }

    int matrixWidth = atoi(argv[2]);
    
    int matrixSize = matrixWidth * matrixWidth;

    float *A, *B, *C, *D;
    cudaMallocManaged(&A, matrixSize * sizeof(float));
    cudaMallocManaged(&B, matrixSize * sizeof(float));
    cudaMallocManaged(&C, matrixSize * sizeof(float));
    cudaMallocManaged(&D, matrixSize * sizeof(float));

    initializeMatrices(A, matrixSize);
    initializeMatrices(B, matrixSize);

    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize(matrixWidth / TILE_WIDTH, matrixWidth / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpu_time = 0;
    float cpu_time = 0;

    cudaEventRecord(start);
    // Launch the kernel
    MatrixMulCUDA<<<gridSize, blockSize>>>(C, A, B, matrixWidth);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&gpu_time, start, stop);

    auto start_time = std::chrono::high_resolution_clock::now();
    // Perform matrix multiplication on CPU and store in D
    matrixMultiplication (A, B, D, matrixWidth);    
    auto end_time = std::chrono::high_resolution_clock::now();
    cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    cpu_time = cpu_time / 1000;

    // ... (Perform matrix multiplication on CPU and store in D)

    // Compare matrices C and D
    bool matricesMatch = compareMatrices(C, D, matrixSize);
    
    if (matricesMatch) {
        printf("SUCCESS!\n");
        printf("CPU Matrix Multiply Time (ms) : %f \n", cpu_time);
        printf("GPU Matrix Multiply Time (ms) : %f \n", gpu_time);
	printf("Speedup: %f \n", cpu_time/gpu_time);
    } else {
        printf("ERROR!\n");
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(D);

    return 0;    
}
