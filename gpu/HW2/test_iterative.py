#!/usr/bin/env python3
"""
Iterative test script for bitonic sort at 100000 level
Runs correctness + performance tests, provides clear metrics for optimization
"""

import subprocess
import re
import os
import sys

TEST_SIZE = 100000  # Test at 100K level
REPEAT = 2  # Run twice for quick iteration
DEBUG_TIMING = False  # Set True when --debug passed

def compile_code(build_dir, bitonic_file, debug_timing=False):
    """Compile the CUDA code. If debug_timing=True, define DEBUG_BITONIC_TIMING for per-phase timing."""
    extra = " -DDEBUG_BITONIC_TIMING" if debug_timing else ""
    compile_cmd = f"nvcc -allow-unsupported-compiler -lineinfo -x cu{extra} {os.path.basename(bitonic_file)} main.cu -o a.out"
    proc = subprocess.Popen(compile_cmd, shell=True, cwd=build_dir, 
                           stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        print(f"[FAIL] Compilation failed:\n{stderr}")
        return False
    print(f"[OK] Compiled: {bitonic_file}" + (" (DEBUG_BITONIC_TIMING)" if debug_timing else ""))
    return True

def run_test(build_dir, size):
    """Run a single test and return output"""
    # Windows-compatible path
    if os.name == 'nt':
        cmd = f"a.out {size}"
    else:
        cmd = f"./a.out {size}"
    proc = subprocess.Popen(cmd, shell=True, cwd=build_dir,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output, error = proc.communicate()
    return output, error

def extract_bitonic_timing(output):
    """Extract [BITONIC TIMING] ... [END BITONIC TIMING] block from output."""
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    clean = ansi_escape.sub('', output)
    start = clean.find('[BITONIC TIMING]')
    end = clean.find('[END BITONIC TIMING]')
    if start == -1 or end == -1:
        return None
    return clean[start:end + len('[END BITONIC TIMING]')]

def extract_metrics(output):
    """Extract performance metrics from output"""
    metrics = {}
    
    # Remove ANSI codes
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    clean_output = ansi_escape.sub('', output)
    
    # Check correctness
    metrics['correct'] = "FUNCTIONAL SUCCESS" in clean_output
    
    # Extract times
    h2d_match = re.search(r"H2D Transfer Time \(ms\):\s*(\d+\.\d+)", clean_output)
    kernel_match = re.search(r"Kernel Time \(ms\)\s*:\s*(\d+\.\d+)", clean_output)
    d2h_match = re.search(r"D2H Transfer Time \(ms\):\s*(\d+\.\d+)", clean_output)
    meps_match = re.search(r"GPU Sort Speed\s*:\s*(\d+\.\d+)", clean_output)
    
    if h2d_match:
        metrics['h2d'] = float(h2d_match.group(1))
    if kernel_match:
        metrics['kernel'] = float(kernel_match.group(1))
    if d2h_match:
        metrics['d2h'] = float(d2h_match.group(1))
    if meps_match:
        metrics['meps'] = float(meps_match.group(1))
    
    # Calculate totals
    if 'h2d' in metrics and 'd2h' in metrics:
        metrics['transfer_total'] = metrics['h2d'] + metrics['d2h']
    if 'kernel' in metrics and 'transfer_total' in metrics:
        metrics['total_time'] = metrics['kernel'] + metrics['transfer_total']
    
    return metrics

def print_summary(results):
    """Print a clear summary of results"""
    print("\n" + "="*70)
    print("ITERATIVE TEST RESULTS (100K elements)")
    print("="*70)
    
    correct_count = sum(1 for r in results if r.get('correct', False))
    print(f"[OK] Correctness: {correct_count}/{len(results)} tests passed")
    
    if correct_count == 0:
        print("[FAIL] All tests failed - fix correctness first!")
        return
    
    # If debug timing was used, show BITONIC TIMING from last run
    if results and DEBUG_TIMING and 'raw_output' in results[-1]:
        timing_block = extract_bitonic_timing(results[-1]['raw_output'])
        if timing_block:
            print("\n" + "="*70)
            print("BITONIC SORT TIMING BREAKDOWN (where the kernel time is spent)")
            print("="*70)
            print(timing_block)
            print("="*70)
    
    # Calculate averages
    h2d_vals = [r['h2d'] for r in results if 'h2d' in r]
    kernel_vals = [r['kernel'] for r in results if 'kernel' in r]
    d2h_vals = [r['d2h'] for r in results if 'd2h' in r]
    transfer_vals = [r['transfer_total'] for r in results if 'transfer_total' in r]
    total_vals = [r['total_time'] for r in results if 'total_time' in r]
    meps_vals = [r['meps'] for r in results if 'meps' in r]
    
    if h2d_vals:
        avg_h2d = sum(h2d_vals) / len(h2d_vals)
        min_h2d = min(h2d_vals)
        print(f"\n[H2D] Transfer:")
        print(f"   Average: {avg_h2d:.3f} ms  |  Best: {min_h2d:.3f} ms")
    
    if kernel_vals:
        avg_kernel = sum(kernel_vals) / len(kernel_vals)
        min_kernel = min(kernel_vals)
        print(f"\n[KERNEL] Time:")
        print(f"   Average: {avg_kernel:.3f} ms  |  Best: {min_kernel:.3f} ms")
    
    if d2h_vals:
        avg_d2h = sum(d2h_vals) / len(d2h_vals)
        min_d2h = min(d2h_vals)
        print(f"\n[D2H] Transfer:")
        print(f"   Average: {avg_d2h:.3f} ms  |  Best: {min_d2h:.3f} ms")
    
    if transfer_vals:
        avg_transfer = sum(transfer_vals) / len(transfer_vals)
        min_transfer = min(transfer_vals)
        print(f"\n[TRANSFER] Total (H2D + D2H):")
        print(f"   Average: {avg_transfer:.3f} ms  |  Best: {min_transfer:.3f} ms")
        print(f"   Target: < 30ms (for 100M), scaled ~0.03ms for 100K")
    
    if total_vals:
        avg_total = sum(total_vals) / len(total_vals)
        min_total = min(total_vals)
        print(f"\n[TOTAL] GPU Time:")
        print(f"   Average: {avg_total:.3f} ms  |  Best: {min_total:.3f} ms")
    
    if meps_vals:
        avg_meps = sum(meps_vals) / len(meps_vals)
        max_meps = max(meps_vals)
        print(f"\n[MEPS] Million Elements Per Second:")
        print(f"   Average: {avg_meps:.2f} MEPS  |  Best: {max_meps:.2f} MEPS")
        print(f"   Target: > 900 MEPS (for 100M), scaled ~0.9 MEPS for 100K")
    
    # Calculate score (scaled from grade.py for 100K)
    # More realistic scaling: 100K is 1000x smaller, but overhead doesn't scale linearly
    # Use relative targets: kernel < 1ms, transfer < 1ms for good score at 100K
    score = 0.0
    max_score = 22.0
    
    # Correctness (5 points)
    if correct_count == len(results):
        score += 5.0
        print(f"\n[SCORE] Correctness: 5.0/5.0 [OK]")
    else:
        print(f"\n[SCORE] Correctness: {correct_count}/5.0")
    
    # Kernel time score (max 10 points)
    # Target: < 1ms for 100K (scaled from 80ms for 100M, but account for overhead)
    if kernel_vals:
        best_kernel = min(kernel_vals)
        target_kernel = 1.0  # Realistic target for 100K
        kernel_score = min(target_kernel / best_kernel * 10, 10) if best_kernel > 0 else 0
        score += kernel_score
        print(f"[SCORE] Kernel Time: {kernel_score:.2f}/10.0 (best: {best_kernel:.3f}ms, target: <{target_kernel:.1f}ms)")
    
    # Transfer time score (max 4 points)
    # Target: < 1ms for 100K
    if transfer_vals:
        best_transfer = min(transfer_vals)
        target_transfer = 1.0  # Realistic target for 100K
        transfer_score = min(target_transfer / best_transfer * 4, 4) if best_transfer > 0 else 0
        score += transfer_score
        print(f"[SCORE] Transfer Time: {transfer_score:.2f}/4.0 (best: {best_transfer:.3f}ms, target: <{target_transfer:.1f}ms)")
    
    # MEPS score (max 14 points)
    # Target: > 50 MEPS for 100K (scaled from 900 for 100M)
    if meps_vals:
        best_meps = max(meps_vals)
        target_meps = 50.0  # Realistic target for 100K
        if best_meps >= target_meps:
            meps_score = min(14, (best_meps / target_meps) * 14)
        else:
            meps_score = (best_meps / target_meps) * 14
        score += meps_score
        print(f"[SCORE] MEPS: {meps_score:.2f}/14.0 (best: {best_meps:.2f}, target: >{target_meps:.0f})")
    
    # Occupancy and throughput (assume max for now, would need ncu for real values)
    print(f"[SCORE] Occupancy: 1.0/1.0 (assumed, need ncu for real)")
    print(f"[SCORE] Memory Throughput: 1.0/1.0 (assumed, need ncu for real)")
    score += 2.0  # Occupancy + throughput
    
    print(f"\n[TOTAL SCORE] {score:.2f}/{max_score:.1f} points")
    
    # Optimization suggestions
    print(f"\n[TIPS] Optimization Suggestions:")
    if kernel_vals and min(kernel_vals) > target_kernel:
        print(f"   - Kernel time is {min(kernel_vals):.3f}ms (target: <{target_kernel:.1f}ms) - try reducing shared mem usage or improving coalescing")
    if transfer_vals and min(transfer_vals) > target_transfer:
        print(f"   - Transfer time is {min(transfer_vals):.3f}ms (target: <{target_transfer:.1f}ms) - ensure pinned memory, consider async")
    if meps_vals and max(meps_vals) < target_meps:
        print(f"   - MEPS is {max(meps_vals):.2f} (target: >{target_meps:.0f}) - focus on reducing total GPU time")
    
    print("="*70 + "\n")
    
    return score

def main():
    global DEBUG_TIMING
    args = [a for a in sys.argv[1:] if a != '--debug']
    if '--debug' in sys.argv:
        DEBUG_TIMING = True
    if len(args) < 1:
        print("Usage: python test_iterative.py <bitonic.cu> [--debug]")
        print("  --debug  compile with DEBUG_BITONIC_TIMING and print per-phase kernel timing")
        sys.exit(1)
    
    bitonic_file = args[0]
    build_dir = os.path.dirname(os.path.abspath(bitonic_file)) or os.getcwd()
    
    print(f"Testing: {bitonic_file}")
    print(f"Test size: {TEST_SIZE:,} elements")
    print(f"Runs: {REPEAT}")
    if DEBUG_TIMING:
        print("Mode: DEBUG (bitonic sort phase timing enabled)")
    
    # Compile
    if not compile_code(build_dir, bitonic_file, debug_timing=DEBUG_TIMING):
        sys.exit(1)
    
    # Run tests
    print(f"\nRunning {REPEAT} tests...")
    results = []
    
    for i in range(REPEAT):
        print(f"  Run {i+1}/{REPEAT}...", end=' ', flush=True)
        output, error = run_test(build_dir, TEST_SIZE)
        metrics = extract_metrics(output)
        if DEBUG_TIMING:
            metrics['raw_output'] = output
        results.append(metrics)
        
        if metrics.get('correct', False):
            print("[OK]", end='')
        else:
            print("[FAIL]", end='')
        print()
        
        if error:
            print(f"    Error: {error[:100]}")
    
    # Print summary and get score
    score = print_summary(results)
    
    # Return success if at least one test passed
    correct_count = sum(1 for r in results if r.get('correct', False))
    return 0 if correct_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
