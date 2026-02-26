#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from typing import Dict, Tuple


POLICY_TO_GPUCFG = {
    "RR": "xmls/gpuconfig_1c_rr.xml",
    "GTO": "xmls/gpuconfig_1c_gto.xml",
    "CCWS": "xmls/gpuconfig_1c_ccws.xml",
}

MACSIM_BIN = "./macsim"
TESTS_JSON = "micro_traces/tests.json"
LOG_DIR = "log/micro_traces"
TIMEOUT_S = 300


def parse_logfile(logfile: str) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    in_stats = False
    with open(logfile, "r", encoding="ascii", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if not in_stats:
                if "============= MacSim Stats =============" in line:
                    in_stats = True
                continue
            if ":" not in line or line.endswith(":"):
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            try:
                stats[key] = float(val)
            except ValueError:
                continue
    return stats


def run_one(macsim_bin: str, gpu_cfg: str, kernel_cfg: str, log_path: str, timeout_s: int) -> Tuple[int, Dict[str, float]]:
    cmd = [macsim_bin, "-g", gpu_cfg, "-t", kernel_cfg]
    with open(log_path, "w", encoding="ascii") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=timeout_s)
    stats = parse_logfile(log_path)
    return proc.returncode, stats


def resolve_kernel_config_path(base_dir: str, kernel_config_entry: str) -> str:
    if os.path.isabs(kernel_config_entry):
        return kernel_config_entry

    normalized = os.path.normpath(kernel_config_entry)
    base_prefix = os.path.normpath(base_dir) + os.sep
    if normalized.startswith(base_prefix):
        return normalized

    return os.path.normpath(os.path.join(base_dir, normalized))


def main() -> int:
    tests_json = TESTS_JSON
    if not os.path.exists(tests_json):
        print(f"Missing tests json: {tests_json}", file=sys.stderr)
        return 2

    os.makedirs(LOG_DIR, exist_ok=True)

    with open(tests_json, "r", encoding="ascii") as f:
        meta = json.load(f)

    tests = meta.get("tests", [])
    if not tests:
        print("No tests found in tests.json", file=sys.stderr)
        return 2

    failed = 0
    for t in tests:
        name = t["name"]
        policy = t["policy"]
        gpu_cfg = POLICY_TO_GPUCFG.get(policy)
        if gpu_cfg is None:
            print(f"[FAIL] {name}: unsupported policy '{policy}'")
            failed += 1
            continue

        kernel_cfg = resolve_kernel_config_path(os.path.dirname(tests_json), t["kernel_config"])
        log_path = os.path.join(LOG_DIR, f"{name}.log")

        try:
            rc, stats = run_one(MACSIM_BIN, gpu_cfg, kernel_cfg, log_path, TIMEOUT_S)
        except subprocess.TimeoutExpired:
            print(f"[FAIL] {name}: timeout ({TIMEOUT_S}s)")
            failed += 1
            continue
        except FileNotFoundError as e:
            print(f"[FAIL] {name}: {e}")
            return 2

        if rc != 0:
            print(f"[FAIL] {name}: macsim exited rc={rc} (log: {log_path})")
            failed += 1
            continue

        expected_cycles = t.get("expected_cycles")
        expected_stalls = t.get("expected_stall_cycles")
        expected_mpki = t.get("expected_mpki")

        got_instr = stats.get("NUM_INSTRS_RETIRED")
        got_mem = stats.get("NUM_MEM_REQUESTS")
        got_cycles = stats.get("NUM_CYCLES")
        got_stalls = stats.get("NUM_STALL_CYCLES")
        got_mpki = stats.get("MISSES_PER_1000_INSTR")

        mismatches = []
        if expected_cycles is not None and (got_cycles is None or int(got_cycles) != int(expected_cycles)):
            mismatches.append(f"NUM_CYCLES expected={expected_cycles} got={got_cycles}")
        if expected_stalls is not None and (got_stalls is None or int(got_stalls) != int(expected_stalls)):
            mismatches.append(f"NUM_STALL_CYCLES expected={expected_stalls} got={got_stalls}")
        if expected_mpki is not None and (got_mpki is None or abs(got_mpki - float(expected_mpki)) > 0.01):
            mismatches.append(f"MISSES_PER_1000_INSTR expected={float(expected_mpki):.2f} got={got_mpki}")

        if mismatches:
            print(f"[FAIL] {name}: " + " ; ".join(mismatches))
            failed += 1
        else:
            print(
                f"[PASS] {name}: "
                f"NUM_CYCLES={int(got_cycles)} "
                f"NUM_STALL_CYCLES={int(got_stalls)} "
                f"NUM_INSTRS_RETIRED={int(got_instr)} "
                f"NUM_MEM_REQUESTS={int(got_mem)} "
                f"MISSES_PER_1000_INSTR={got_mpki:.2f}"
            )

    total = len(tests)
    passed = total - failed
    print(f"\nSummary: {passed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
