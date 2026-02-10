# CS 7295 GPU HW & SW - Project 2

## Objectives
The objective of this assignment is to further advance your understanding of parallel
programming using CUDA. You will write CUDA code to perform parallel integer sorting
and gain a deeper understanding of parallelism in GPU programming. This assignment
enhances your CUDA programming skills and provides an exercise to get familiar with
the NSight Compute profiler.

The goal of this assignment is to implement parallel sorting for arrays with the assistance
of the NSight profiler and achieve high performance on the NVIDIA H100 GPU.

## Instructions
### Compiling and Running Code
Input to the program is a 1-D array. You have been provided with the initial framework for
the code in this Git repository.

- Compile the code
`make all`

- Run an individual array (example: array size 10K)
`./a.out 10000`

>To Debug code: `make debug` will compile with flags -g -G to allow use of cuda-gdb.

#### Using the Grading Script
Once your code is fuctional, you can run the grading script, `grade.py`, to see your code's performance and the score you would receive. The grading script assumes you will submit a satisfactory report and includes the one point for for the report in the score. 

Running the grading script:
```
python grade.py bitonic.cu
```
> For development purposes, the grading script will accept any name for the bitonic.cu file.  Your submission to GS will require you to submit bitonic.cu and student.h as noted below.

### Writing Code
You will modify two files to implement your solution.

- bitonic.cu - this file will contain:
    - your CUDA kernel(s), 
    - implementation of the wrapper functions main.cu will call
    - helper methods 
- student.h - student modifiable values and a place for storing any custom values you create

>bitonic.cu and student.h will be the only two files submitted for grading. Therefore, do not modify any other files as part of your solution.

>The other files provided from the GitHub repository are the same files that will be used in the grading environment.

## Background
The straightforward implementation of merge sort on a GPU can exhibit suboptimal runtime due to the nature of the algorithm. As each iteration reduces the active threads by half and the last iteration involves merging the entire array, it leads to inefficient parallelization. This reduction in active threads hinders the GPU's ability to fully exploit its
parallel processing capabilities. As an exercise (not required for the assignment), you can write a CUDA program to perform a straightforward parallelization of the mergesort algorithm using <<< N, M >>> kernels. What does the ‘Achieved Occupancy’ look like for kernel launches in the later iterations on NSight?

Divide and conquer, an effective paradigm for parallel algorithms, involves breaking a problem into smaller subproblems solved recursively, enabling concurrent processing. Mergesort, an optimal sequential sorting algorithm utilizing divide and conquer, serves as inspiration for parallel sorting algorithms like Bitonic sort. Bitonic sort efficiently maintains parallelism, making it well-suited for GPU architectures. Another notable approach is Batcher's odd-even merge sort, leveraging a sorting network for effective
parallelism in sorting operations.

## Bitonic Sort Explained 
Bitonic Sort creates a bitonic sequence, which is a sequence that starts as ascending and then becomes descending (or vice versa). The algorithm recursively sorts subsequences of the bitonic sequence until the entire sequence is sorted. It works on sequences with lengths that are powers of 2.

**Bitonic split**: of a bitonic sequence L = x0, x1, x2 .. xn-1 is defined as decomposition of L into:
    
    Lmin = min(x0, xn/2), min(x1, x(n/2)+1) . . . min(x(n/2)-1, xn-1)
    
    Lmax = max(x0, xn/2), max(x1, x(n/2)+1) . . . max(x(n/2)-1, xn-1)

Lmin and Lmax are also bitonic sequences with max(Lmin) ≤ min(Lmax)

**Bitonic merge**: Turning a bitonic sequence into a sorted sequence using repeated bitonic
split operations. Successive bitonic split operations are applied on the decomposed subsequences until their size reduces to 2, culminating in a fully sorted sequence.

    BM(n) = BS(n) + BS(n/2) + . . BS(2) = O(log n)

To achieve a bitonic sequence, we start with a sequence of length 2 and apply Bitonic merge to obtain bitonic sequences of length 2. The process involves alternating between ascending and descending order to transform an unsorted array into a bitonic sequence. This serves as the initial step before applying further Bitonic merge operations to eventually achieve a fully sorted sequence.

    BitonicSort(n) = BM(2) + BM(4) + . . BM(n) = O(log^2 n)

 ![Bitonic Sort from https://hwlang.de/algorithmen/sortieren/bitonic/bitonicen.htm](https://hwlang.de/algorithmen/sortieren/bitonic/binetzen.gif)



## Task #1
You need to implement bitonic sort in CUDA. Pseudo code for bitonic sort algorithm to sort n elements:

```
for i=1 to (log n) do
    for j=i-1 down to 0 do
        for k=0 to n do #loop through the array
            a = arr[k]; b = arr[k XOR 2^j];
            if (k XOR 2^j) > k then # (a,b) are compared so skip (b,a) case
                if (2^i & k) is 0 then
                    Compare_Exchange↑ with (a, b)
                else
                    Compare_Exchange↓ with (a, b)
        endfor
    endfor
endfor
```

The outer loop sequentially traverses the stages of the Bitonic sort algorithm. Each iteration corresponds to the execution of the BM(2^i) operation, resulting in the generation of sorted sequences with lengths of 2^i. The inner loop performs multiple bitonic splits essential for completing a bitonic merge operation.

The **compare exchange** swaps elements. The sorting strategy ( (2^i & k) = 0 check) arranges even chunks (sub-sequences of length 2^i) in ascending order and odd chunks in descending order, forming a 2^{i+1} bitonic sequence for the subsequent iteration.

```
Compare_Exchange↑: if (arr[i] > arr[j])
                        arr[i], arr[j] = arr[j], arr[i]
```

The XOR operation determines the indices of the two elements to be compared during the jth iteration. For instance, if i = 2 and j = 2 (first sub-stage of BM(8)), XORing 0 with 4 yields 4, 1 gives 5, and so forth. To gain a better understanding, it is encouraged to experiment with various values of i and j and compare your results with the image in the previous page. In simpler terms, XOR strides the rank by 2^j, providing the indices required for comparing elements during the algorithm's iterations.

## Task #2
### CUDA Optimizations
We move on to optimizing our parallel program using our learnings from lectures and NSight profiler observations to improve memory and compute efficiency. The method described in Task#1 leads to excessive kernel launches and prolonged global memory access times. NSight may indicate:

>This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device.”

Shared memory to rescue: If a subsequence of size 2^s fits into the shared memory of a block, process steps s,s-1 . . .1 of the bitonic sort on the shared memory to reduce global accesses and also kernel launches. Break down your kernel into two: one with the shared memory and another with the global memory.

For the performance evaluation, we will run your solution with a 100M (100,000,000) element array size. We will evaluate a few times and take the best run to avoid any server load issues. We will run your code on the H100 in PACE. Be sure to explicitly select the H100 when creating an instance for you final performance testing runs.

Please try to optimize the programs to make # of million elements per second (meps) as high as possible. You will get full points when achieving 900 million elements per second.

To encourage you to get familiar with the NVIDIA profiling toolkit, we also evaluate the ‘Memory Throughput’ and ‘Achieved Occupancy’. If you have multiple kernels in your implementation, we will take the average of all the kernels. You are expected to get Memory Throughput higher than 80% and ‘Achieved Occupancy’ higher than 70% on an H100 GPU.

You can use NVIDIA ncu to get these metric:

#### Memory Throughput
```
ncu --metric
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
--print-summary per-gpu a.out 10000000
```

#### Achieved Occupancy
```
ncu --metric sm__warps_active.avg.pct_of_peak_sustained_active --
print-summary per-gpu a.out 10000000
```

## What to submit:
You will upload two files to GradeScope. Do not create a zip file containing these two files. The files must have the following names or point deductions will be applied.

- report.pdf – The report must be a .pdf file to receive credit.
- submission.zip (This file can be built with the make-submission.sh script that is part of the project frame.)


### What will Gradescope do and how do I get my grade?
Gradescope will do two verification steps:
    
- Verify that you submitted a report.pdf file and a submission.zip file.

- Verify that the code you wrote in bitonic.cu is in the correct locations within the file.

>Gradescope will NOT run and grade your code. It isn’t capable of doing this since we are using GPUs on the project. Gradescope will not “read” your report.

After the project closes, your code will be run on the ICE cluster multiple times to get you the best grade possible. The teaching team will read the reports and give appropriate credit. The final grade will be entered into Canvas.

## Grading Policy

**Total points: 20 plus up to 2 extra points based on meeting performance metrics**

The program will be graded on the correctness and usage of the required parallel programming features. You should also use a good programming style and add comments to your program so that it is understandable.

The script used to grade is provided (grade.py) to evaluate the score locally. All submissions are evaluated using H100 GPUs in the PACE-ICE cluster. Be sure to explicitly select the H100 when creating an instance.

>Start the assignment early to circumvent a last-minute rush to secure a node on the pace-ice cluster.

Grading consists of the following components:

1. Functional correctness (5 pts)

We will check whether the sorting results match the expected results for the input array of sizes 2K, 10K, 100K, 1M, 10M.

Each test case of array size will give 1 pts. “FUNCTIONAL SUCCESS” is printed on the terminal for passing cases.

If your code is not parallel code, you will get only 20% of functional correctness (i.e., 1 pt)

2. Performance (16 pts)

We will go through your code to make sure the appropriate parallel programming practices discussed in the class are being followed along with the right CUDA functions being called. The evaluation metrics will be run for 100M array size. Please make sure your implementation is robust at this scale. Note: off-loading compute operations (e.g. final ‘merge’) to CPU is not allowed.

>Note: multi-threaded host code is not permitted.

>Note: You can only get performance points when your implementation is
functionally correct.

|Evaluation metric | Max Credit | Calculation             |
|----------------- | ---------- | ------------------------|
| Achieved Occupancy | 1 | Achieved Occupancy >= 65% |
| Memory Throughput | 1 | Memory Throughput >= 75%|
| **Performance Option 1** |
|Million elements per second (meps) | 14 | if meps > 900 min( (meps/1000)*14, 14)
| **Performance Option 2** |
| Kernel time (ms) | 10 | min( (80/kernelTime)*10, 10) |
| Memory Transfer Time D2H + H2D | 4 | min((30/memTime)*4, 4) |

>Note: Performance Options (14 points)

- Option 1 is million elements processed per second. If your meps score is
900 or over, then you are eligible for this score. Full credit is obtained with a meps score of 1000.
- Option 2 is a combination of time spent in the GPU kernel doing calculations
where 80ms or less is ideal and full credit of 10 points is given and time spent transporting data to and from the GPU where 30ms or less is ideal and full credit of 4 points is given.
- You will receive the higher score of the two options.

> Note: Points will be deducted for ignoring performance protocols; **serialization in the program will lead to a zero on the whole assignment.**

3. Report (1 pt)

When describing the project, what you did, what you tested, etc., you can assume that the reader is familiar with the topic. Also, be sure to refer to the syllabus for expectations on Assignment Quality.

Report contents:

- Your report will discuss project implementation and performance counter
analysis
- Discussion of performance optimization techniques implemented and
effectiveness of each
- Report length must be between two and three pages (including charts and
graphs.) You will use 11 pt. Times New roman or Arial fonts. These rules
are in place so that you write enough (but no too much) in your report.

>You must submit a report to receive a grade. Submissions without a report
will earn zero points for this project. Remember that this is a graduate level
course and graduate level work is expected.

## NSight Compute

Profilers are tools that sample and measure performance characteristics of an
executable across its runtime. This information is intended to aid program optimization and performance engineering. Nsight Compute - Provides an in depth level assessment of individual GPU kernel performance and how various GPU resources are utilized across many different metrics. Use NSight Compute NvProf to report the following numbers as well:

- Number of global memory accesses
- Number of local memory accesses
- Number of divergent branches
- Achieved Occupancy

This YouTube video can serve as a good starting point to using the profiler visualizations to optimize code: https://developer.nvidia.com/nsight-compute

Running NSight Compute:

    ncu ./<program name> # stats for each kernel on stdout
    ncu -o profile ./<program name> # output file for NSight Compute GUI

Command to list all existing metrics:

    ncu --query-metrics --query-metrics-mode all

To check the metrics you can use this command

    ncu --metrics [metric_1],[metric_2],... ./\<program name>

You can read ncu documentation to understand what each metric means. Here are
some metrics that you probably need:

    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
    l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum

    smsp__thread_inst_executed_per_inst_executed.ratio
    smsp__thread_inst_executed_pred_on_per_inst_executed.ratio

    sm__maximum_warps_per_active_cycle_pct
    sm__warps_active.avg.pct_of_peak_sustained_active
    
## Additional Resources

- [Bitonic Sort](https://hwlang.de/algorithmen/sortieren/bitonic/bitonicen.htm)
- [Prof. Vuduc’s bitonic sort lecture: items 23 (Comparator networks) through 28](https://youtube.com/playlist?list=PLAwxTw4SYaPk8NaXIiFQXWK6VPnrtMRXC&si=3de_ddjnULgEqV6D)
- [Batcher’s Odd-Even Merge Sort](https://hwlang.de/algorithmen/sortieren/networks/oemen.htm)
- [Improved GPU Sorting](https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting)
- [PACE ICE cluster guide](https://docs.pace.gatech.edu/ice_cluster/ice/)
- [NVIDIA CUDA Toolkit Documentation](https://developer.nvidia.com/cuda-toolkit)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)


Version: December 17, 2025