#!/usr/bin/env python3
"""Benchmark GPU verifier phase latency on synthetic eBPF programs."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


DEFAULT_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]
DEFAULT_RUNS = 50
DEFAULT_TARGET = "bpftime_gpu_verify_perf"
RESULT_BASENAME = "perf_breakdown_results"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_BUILD_DIR = REPO_ROOT / "build"
DEFAULT_BINARY = DEFAULT_BUILD_DIR / "bpftime-verifier" / DEFAULT_TARGET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GPU verifier latency on synthetic eBPF programs."
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=DEFAULT_BINARY,
        help=f"Path to the perf binary (default: {DEFAULT_BINARY})",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help=f"CMake build directory for auto-build (default: {DEFAULT_BUILD_DIR})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Number of runs per program size (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Instruction counts to benchmark",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR,
        help=f"Output directory for markdown/JSON/CSV (default: {SCRIPT_DIR})",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Fail instead of auto-building the perf binary when it is missing",
    )
    return parser.parse_args()


def ensure_binary(binary: Path, build_dir: Path, no_build: bool) -> None:
    if binary.exists():
        return

    if no_build:
        raise FileNotFoundError(
            f"Verifier perf binary not found: {binary}. Build {DEFAULT_TARGET} first."
        )

    if not (build_dir / "CMakeCache.txt").exists():
        raise FileNotFoundError(
            f"Build directory is not configured: {build_dir}. "
            "Run CMake with -DENABLE_EBPF_VERIFIER=YES first."
        )

    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", DEFAULT_TARGET],
        cwd=REPO_ROOT,
        check=True,
    )

    if not binary.exists():
        raise FileNotFoundError(
            f"Built target {DEFAULT_TARGET}, but binary is still missing at {binary}."
        )


def run_once(binary: Path, program_size: int) -> dict[str, float]:
    completed = subprocess.run(
        [str(binary), "--size", str(program_size)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    if payload["program_size"] != program_size:
        raise RuntimeError(
            f"Binary reported size {payload['program_size']} for requested size {program_size}."
        )
    return {
        "size": int(payload["program_size"]),
        "total_time_us": float(payload["total_time_us"]),
        "prevail_time_us": float(payload["prevail_time_us"]),
        "simt_time_us": float(payload["simt_time_us"]),
    }


def benchmark(binary: Path, sizes: list[int], runs: int) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    for size in sizes:
        samples = [run_once(binary, size) for _ in range(runs)]
        total_median = statistics.median(sample["total_time_us"] for sample in samples)
        prevail_median = statistics.median(
            sample["prevail_time_us"] for sample in samples
        )
        simt_median = statistics.median(sample["simt_time_us"] for sample in samples)
        results.append(
            {
                "size": size,
                "total_time_us": round(total_median, 3),
                "prevail_time_us": round(prevail_median, 3),
                "simt_time_us": round(simt_median, 3),
            }
        )
    return results


def make_markdown(results: list[dict[str, float]]) -> str:
    lines = [
        "| Size | Total (μs) | SIMT Pass (μs) |",
        "| ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {int(row['size'])} | {row['total_time_us']:.3f} | "
            f"{row['simt_time_us']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path, results: list[dict[str, float]]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = out_dir / f"{RESULT_BASENAME}.md"
    json_path = out_dir / f"{RESULT_BASENAME}.json"
    csv_path = out_dir / f"{RESULT_BASENAME}.csv"

    markdown_path.write_text(make_markdown(results), encoding="utf-8")
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    csv_lines = ["size,total_time_us,simt_time_us"]
    for row in results:
        csv_lines.append(
            f"{int(row['size'])},{row['total_time_us']:.3f},"
            f"{row['simt_time_us']:.3f}"
        )
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    return {
        "markdown": markdown_path,
        "json": json_path,
        "csv": csv_path,
    }


def main() -> int:
    args = parse_args()
    ensure_binary(args.binary, args.build_dir, args.no_build)
    results = benchmark(args.binary, args.sizes, args.runs)
    paths = write_outputs(args.out_dir, results)

    print(make_markdown(results), end="")
    print(f"Wrote markdown: {paths['markdown']}")
    print(f"Wrote JSON: {paths['json']}")
    print(f"Wrote CSV: {paths['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
