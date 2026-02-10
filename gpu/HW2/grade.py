# grade.py
import subprocess
import re
import sys
import uuid
import os
import shutil
import argparse

REPEAT = 3

# The GPU architecture settings will be passed dynamically based on detected GPU
def set_gpu_architectures(GPU_arch):
    global lower_bound_occupancy, lower_bound_throughput

    lower_bound_occupancy = 65
    lower_bound_throughput = 75

def compile(build_dir, bitonic_file, with_arch_flag=False, sm_version=None):
    compile_command = f"nvcc -Xcompiler -rdynamic -lineinfo -x cu {os.path.basename(bitonic_file)} main.cu -o a.out"
    if with_arch_flag and sm_version:
        compile_command += f" -arch=sm_{sm_version}"
    compile_process = subprocess.Popen(compile_command, shell=True, cwd=build_dir, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = compile_process.communicate()
    compile_return_code = compile_process.returncode
    if compile_return_code != 0:
        error_message = stderr.strip()
        print(f"Compilation failed: {error_message}")
        return False
    print(f"Compiled! {bitonic_file}") 
    return True

def run(build_dir, size):
    run_command = f"./a.out {size}"
    run_process = subprocess.Popen(run_command, shell=True, cwd=build_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    run_output, run_error = run_process.communicate()
    if "FUNCTIONAL SUCCESS" in run_output:
        return run_output
    print(f"Functional test failed for size = {size}")
    return None

def run_ncu_metric(build_dir, metric, size):
    ncu_command = f"ncu --metric {metric} --print-summary per-gpu ./a.out {size}"
    ncu_process = subprocess.Popen(ncu_command, shell=True, cwd=build_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    ncu_output, _ = ncu_process.communicate()
    last_numbers = [float(match.group(3)) for match in re.finditer(rf"{metric}.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)", ncu_output)]
    return last_numbers

def calculate_average_metric(metric_values):
    if metric_values:
        average = sum(metric_values) / len(metric_values)
        return round(average, 2)
    return None

def main(build_dir, with_arch_flag=True, sm_version=None, GPU_arch=None, perf_only=False):
    set_gpu_architectures(GPU_arch)
    
    # Initialize scores
    total_score = 0
    correctness_score = 0
    occupancy_score = 0
    memory_throughput_score = 0
    kernel_time_score = 0
    memory_transfer_score = 0
    meps_score = 0
    report_score = 1  # Default report score

    # Remove ANSI escape sequences from output
    def remove_ansi_escape_sequences(text):
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        return ansi_escape.sub('', text)

    # Run full test suite or just performance
    if perf_only is True:
        # Interested in only the meps / performance scores
        ao = 0
        mt = 0
        report_score = 0
        # Assume correctness
        correctness_score = 5

    else: 
        # Check functional correctness (max 5 points)
        for size in [2000, 10000, 100000, 1000000, 10000000]:
            output = run(build_dir, size)
            if output:
                correctness_score += 1

        # Check achieved occupancy (max 1 point)
        metric_output = run_ncu_metric(build_dir, "sm__warps_active.avg.pct_of_peak_sustained_active", 10000000)
        ao = calculate_average_metric(metric_output)
        print(f"Achieved Occupancy: {ao}")
        if ao is not None and ao >= lower_bound_occupancy:
            occupancy_score = 1

        # Check memory throughput (max 1 point)
        metric_output = run_ncu_metric(build_dir, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", 10000000)
        mt = calculate_average_metric(metric_output)
        print(f"Memory Throughput: {mt}")
        if mt is not None and mt >= lower_bound_throughput:
            memory_throughput_score = 1

    # Start performance Testing
    best_kernel_time = float('inf')
    best_mem_transfer_time = float('inf')

    perf_test_size = 100_000_000  # Use larger size for performance test
    for _ in range(REPEAT):
        output = run(build_dir, perf_test_size)
        if output:
            output = remove_ansi_escape_sequences(output)
            
            # First check if performance passed
            if "PERF FAILING" in output:
                continue  # Skip failed performance runs

            print(output)
            # Extract kernel time (gpuTime)
            kernel_match = re.search(r"Kernel Time \(ms\)\s*:\s*(\d+\.\d+)", output)
            if kernel_match:
                kernel_time = float(kernel_match.group(1))
                best_kernel_time = min(best_kernel_time, kernel_time)

            # Extract H2D and D2H times
            h2d_match = re.search(r"H2D Transfer Time \(ms\):\s*(\d+\.\d+)", output)
            d2h_match = re.search(r"D2H Transfer Time \(ms\):\s*(\d+\.\d+)", output)
            if h2d_match and d2h_match:
                total_mem_time = float(h2d_match.group(1)) + float(d2h_match.group(1))
                best_mem_transfer_time = min(best_mem_transfer_time, total_mem_time)

    # Calculate kernel time score (max 10 points, ideal: 8ms)
    if best_kernel_time != float('inf'):
        kernel_time_score = min(80/best_kernel_time * 10, 10)
        print(f"Kernel Time: {best_kernel_time}ms, Score: {round(kernel_time_score,3)}")

    # Calculate memory transfer time score (max 4 points, ideal: 3ms)
    if best_mem_transfer_time != float('inf'):
        memory_transfer_score = min(30/best_mem_transfer_time * 4, 4)
        print(f"Memory Transfer Time: {best_mem_transfer_time}ms, Score: {round(memory_transfer_score,3)}")

    total_gpu_time = best_kernel_time + best_mem_transfer_time
    meps = perf_test_size / 1e6 / (total_gpu_time * 0.001)
    if meps > 900:
        meps_score = min(14, (meps/1000)*14)
    print(f"Million elements per second: {round(meps,3)}")

    # Calculate total score
    if correctness_score == 5:  # Only add performance scores if all functional tests pass
        total_score = (correctness_score + occupancy_score + memory_throughput_score + max(memory_transfer_score + 
                      kernel_time_score, meps_score) + report_score)
    else:
        total_score = correctness_score + report_score

    total_score = round(total_score, 2)
    print(f"Total Score: {total_score} pts")

    return ao, mt, meps, best_kernel_time, best_mem_transfer_time, correctness_score, occupancy_score, memory_throughput_score, kernel_time_score, memory_transfer_score, meps_score, report_score, total_score

def grade_file(bitonic_file, perf_only=False, with_arch_flag=False, sm_version=None, GPU_arch=None, build_dir=None):
    # Use build_dir if provided, otherwise use current working directory
    if build_dir is None:
        build_dir = os.getcwd()
    try:
        if not compile(build_dir, bitonic_file, with_arch_flag, sm_version):
            return {
                "achieved_occupancy": None,
                "memory_throughput": None,
                "million_elements_per_second": None,
                "kernel_time_ms": None,
                "memory_transfer_time_ms": None,
                "correctness_score": 0,
                "occupancy_score": 0,
                "memory_throughput_score": 0,
                "kernel_time_score": 0,
                "memory_transfer_score": 0,
                "meps_score": 0,
                "report_score": 0,
                "total_score": 0,
                "error_message": "Compilation failed"
            }
        results = main(build_dir, with_arch_flag, sm_version, GPU_arch, perf_only=perf_only)
        (ao, mt, meps, best_kernel_time, best_mem_transfer_time, correctness_score, occupancy_score, memory_throughput_score, kernel_time_score, memory_transfer_score, meps_score, report_score, total_score) = results
        return {
            "achieved_occupancy": ao,
            "memory_throughput": mt,
            "million_elements_per_second": meps,
            "kernel_time_ms": best_kernel_time,
            "memory_transfer_time_ms": best_mem_transfer_time,
            "correctness_score": correctness_score,
            "occupancy_score": occupancy_score,
            "memory_throughput_score": memory_throughput_score,
            "kernel_time_score": kernel_time_score,
            "memory_transfer_score": memory_transfer_score,
            "meps_score": meps_score,
            "report_score": report_score,
            "total_score": total_score,
            "error_message": None
        }
    except Exception as e:
        return {
            "achieved_occupancy": None,
            "memory_throughput": None,
            "million_elements_per_second": None,
            "kernel_time_ms": None,
            "memory_transfer_time_ms": None,
            "correctness_score": 0,
            "occupancy_score": 0,
            "memory_throughput_score": 0,
            "kernel_time_score": 0,
            "memory_transfer_score": 0,
            "meps_score": 0,
            "report_score": 0,
            "total_score": 0,
            "error_message": f"Error during grading: {str(e)}"
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grade a single CUDA submission.")
    parser.add_argument("bitonic_file", nargs="?", help="The bitonic.cu file to grade.")
    parser.add_argument("--perf-only", action="store_true", help="Only run performance tests.")
    parser.add_argument("--build-dir", type=str, default=None, help="Directory to use for build files.")
    args = parser.parse_args()

    if not args.bitonic_file:
        print("Usage: python grade.py <bitonic.cu> [--perf-only] [--build-dir <directory>]")
        exit(1)

    result = grade_file(args.bitonic_file, perf_only=args.perf_only, build_dir=args.build_dir)