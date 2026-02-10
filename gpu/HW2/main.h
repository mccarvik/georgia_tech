/**
 * This code may not be modified.  You will not be submitting this file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>

#include "student.h"

// Define array to be sorted and the size variable
extern DTYPE *arrCpu;
extern int size;

// Pointer to the array returned from the GPU
extern DTYPE *arrSortedGpu;


// Main CUDA program functions

// Function to transfer data from host to device using the student's
// preferred approach
void host_to_dev();

// Function to perform the Bitonic sort and merge functions This function
// will call the kernel(s)
void bitonic_sort();

// Function to return data from Device to Host using the student's 
// preferred approach.
DTYPE *dev_to_host();

// Function to perform cleanup memory and anythong else the student's
// preferred approach requires.
void cleanup();

