#!/usr/bin/env python3
"""Trap uprobe backend microbenchmark driver.

Runs the trap-uprobe-bench binary multiple times, collects statistics,
and writes results to results.md.  Works on native riscv64 and under
qemu-user (pass --qemu to enable).

Usage:
    python3 benchmark/trap-uprobe/benchmark.py [--iter N] [--runs N] [--threads N] [--qemu]
"""
import argparse
import math
import os
import pathlib
import platform
import re
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def find_bench_binary():
    candidates = [
        PROJECT_ROOT / "build-rv" / "benchmark" / "trap-uprobe" / "trap-uprobe-bench",
        PROJECT_ROOT / "build" / "benchmark" / "trap-uprobe" / "trap-uprobe-bench",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def parse_output(text: str) -> dict:
    result = {}
    for m in re.finditer(
        r"Benchmarking\s+(\S+).*?\n\s*Average time usage\s+([\d.]+)\s+ns",
        text,
    ):
        result[m.group(1)] = float(m.group(2))
    # multi-thread lines
    for m in re.finditer(
        r"Thread\s+(\d+):\s+Average time usage\s+([\d.]+)\s+ns", text
    ):
        key = f"thread_{m.group(1)}"
        result[key] = float(m.group(2))
    return result


def stats(values):
    avg = sum(values) / len(values)
    variance = sum((x - avg) ** 2 for x in values) / len(values)
    return {
        "min": min(values),
        "max": max(values),
        "avg": avg,
        "std_dev": math.sqrt(variance),
    }


def run_benchmark(binary, iterations, threads, runs, use_qemu):
    cmd = []
    if use_qemu:
        sysroot = os.environ.get("QEMU_LD_PREFIX", "/usr/riscv64-linux-gnu")
        cmd = ["qemu-riscv64", "-L", sysroot]
    cmd += [str(binary), str(iterations), str(threads)]

    collected = {}
    for i in range(runs):
        print(f"  run {i+1}/{runs}...", end=" ", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            print(f"FAILED (rc={proc.returncode})")
            print(proc.stderr)
            sys.exit(1)
        parsed = parse_output(proc.stdout)
        print(f"uprobe={parsed.get('__trap_bench_uprobe', '?'):.1f}ns")
        for k, v in parsed.items():
            collected.setdefault(k, []).append(v)
    return {k: stats(v) for k, v in collected.items()}


def generate_report(results, args, elapsed):
    lines = [
        "# Trap uprobe backend benchmark results\n",
        f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
        "## Environment\n",
        f"- **Platform:** {platform.machine()} ({'qemu-user' if args.qemu else 'native'})",
        f"- **Kernel:** {platform.release()}",
        f"- **Build:** Release",
        f"- **Iterations per run:** {args.iter}",
        f"- **Runs:** {args.runs}",
        f"- **Threads:** {args.threads}",
        f"- **Total time:** {elapsed:.1f}s",
        "",
        "## Results (ns/call)\n",
        "| Probe type | Min | Avg | Max | Std Dev |",
        "|---|---:|---:|---:|---:|",
    ]
    probe_order = [
        ("baseline", "Baseline (no probe)"),
        ("__trap_bench_uprobe", "Uprobe"),
        ("__trap_bench_uretprobe", "Uretprobe"),
        ("__trap_bench_uprobe_uretprobe", "Uprobe + Uretprobe"),
    ]
    for key, label in probe_order:
        if key in results:
            s = results[key]
            lines.append(
                f"| {label} | {s['min']:.2f} | {s['avg']:.2f} "
                f"| {s['max']:.2f} | {s['std_dev']:.2f} |"
            )

    # overhead
    if "baseline" in results and "__trap_bench_uprobe" in results:
        base = results["baseline"]["avg"]
        lines += [
            "",
            "## Overhead (ns, avg probe - avg baseline)\n",
            "| Probe type | Overhead (ns) |",
            "|---|---:|",
        ]
        for key, label in probe_order[1:]:
            if key in results:
                lines.append(
                    f"| {label} | {results[key]['avg'] - base:.2f} |"
                )

    # multi-thread
    thread_keys = sorted(
        [k for k in results if k.startswith("thread_")],
        key=lambda k: int(k.split("_")[1]),
    )
    if thread_keys:
        lines += [
            "",
            f"## Multi-thread ({len(thread_keys)} threads, uprobe+uretprobe)\n",
            "| Thread | Min | Avg | Max | Std Dev |",
            "|---|---:|---:|---:|---:|",
        ]
        for k in thread_keys:
            s = results[k]
            tid = k.split("_")[1]
            lines.append(
                f"| {tid} | {s['min']:.2f} | {s['avg']:.2f} "
                f"| {s['max']:.2f} | {s['std_dev']:.2f} |"
            )

    if args.qemu:
        lines += [
            "",
            "## Notes\n",
            "These numbers were collected under qemu-user, which adds significant",
            "overhead to signal delivery.  Multiply by roughly 0.05-0.1x to estimate",
            "native riscv64 hardware performance (based on the README's 5.5/8.2 us",
            "measurements on a 32-core board).",
        ]

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Trap uprobe benchmark")
    parser.add_argument("--iter", type=int, default=50000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--qemu", action="store_true")
    args = parser.parse_args()

    binary = find_bench_binary()
    if binary is None:
        print("trap-uprobe-bench binary not found; build it first", file=sys.stderr)
        sys.exit(1)
    print(f"Binary: {binary}")
    print(f"Config: iter={args.iter} runs={args.runs} threads={args.threads} qemu={args.qemu}")

    t0 = time.time()
    results = run_benchmark(binary, args.iter, args.threads, args.runs, args.qemu)
    elapsed = time.time() - t0

    report = generate_report(results, args, elapsed)
    out_path = SCRIPT_DIR / "results.md"
    out_path.write_text(report)
    print(f"\nResults written to {out_path}")
    print(report)


if __name__ == "__main__":
    main()
