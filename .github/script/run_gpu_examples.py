#!/usr/bin/env python3
"""Build and run every GPU example in one job, one after another.

The GPU examples used to run one job per example, one at a time on the
GPU runner, and each job spent about four minutes downloading the
bpftime build artifact around about 40 seconds of build and test. This
runs them in the job that built bpftime, so nothing is uploaded or
downloaded. Each example is built, then run and checked by
run_gpu_example.sh; its logs go to gpu-example-logs/<name>/. Each
example is reported in its own log group and in the job summary, and the
job fails when any example fails.
"""
import os
import shutil
import subprocess
import sys
import time

EXAMPLES = [
    dict(name="atomizer", path="atomizer", executable="./atomizer", victim="./vec_add", expected="Exited _Z9vectorAdd", mode="two-process"),
    dict(name="kernel_trace", path="kernel_trace", executable="./kernel_trace", victim="./vec_add", expected="total events:", mode="two-process"),
    dict(name="gpu_shared_map", path="gpu_shared_map", executable="./gpu_shared_map", victim="./vec_add", expected="counter[0]=", mode="two-process"),
    dict(name="threadscheduling", path="threadscheduling", executable="./threadscheduling", victim="./vec_add", expected="SM Utilization Histogram", mode="two-process"),
    dict(name="threadscheduling_dynamic_hook", path="threadscheduling_dynamic_hook", executable="./threadscheduling", victim="./vec_add 4 64", expected="mapping probe started", mode="dynamic-attach"),
    dict(name="cudagraph", path="cudagraph", executable="./cuda_probe", victim="./vec_add_graph", expected="calls:", mode="two-process"),
    dict(name="threadhist-gpu-kernel-shared-map", path="threadhist-gpu-kernel-shared-map", executable="./threadhist", victim="./vec_add", expected="value(0)=", mode="two-process"),
    dict(name="directly_run_on_gpu", path="directly_run_on_gpu", executable="./directly_run", victim="", expected="GEMM C[0]=", mode="direct-run"),
    dict(name="kernelretsnoop", path="kernelretsnoop", executable="./kernelretsnoop", victim="./vec_add", expected="Thread (", mode="two-process"),
    dict(name="threadhist", path="threadhist", executable="./threadhist", victim="./vec_add", expected="Thread ", mode="two-process"),
    dict(name="cuda-counter", path="cuda-counter", executable="./cuda_probe", victim="./vec_add", expected="calls:", mode="two-process"),
    dict(name="launchlate", path="launchlate", executable="./launchlate", victim="./vec_add", expected="Monitoring CUDA kernel launch latency", mode="two-process"),
    dict(name="launchlate-kernel-gpu-shared-map", path="launchlate-kernel-gpu-shared-map", executable="./launchlate", victim="./vec_add", expected="Monitoring CUDA kernel launch latency", mode="two-process"),
    dict(name="mem_trace", path="mem_trace", executable="./mem_trace", victim="./vec_add", expected="counter[0]=", mode="two-process"),
    dict(name="host_map_test", path="host_map_test", executable="./host_map_test", victim="./vec_add", expected="(call_count)", mode="two-process"),
    dict(name="cutlass", path="cutlass", executable="./counter/cutlass_launch_counter", victim="./gemm/cutlass_gemm --shape 1024x1024x1024 --launches 4 --warmup 1 --verify", expected="CUTLASS launches:", mode="two-process"),
]


def main() -> int:
    root = os.getcwd()
    env = dict(os.environ)
    if shutil.which("clang") is None and shutil.which("clang-18") is not None:
        env["CLANG"] = "clang-18"
    # Some examples (e.g. CUTLASS) use CUDA_HOME/NVCC directly instead of
    # PATH lookups.
    env["CUDA_HOME"] = env.get("BPFTIME_CUDA_ROOT") or "/usr/local/cuda"
    env["NVCC"] = os.path.join(env["CUDA_HOME"], "bin", "nvcc")
    logs = os.path.join(root, "gpu-example-logs")
    os.makedirs(logs, exist_ok=True)

    built, results = set(), []
    for ex in EXAMPLES:
        name, path = ex["name"], ex["path"]
        print(f"::group::{name}", flush=True)
        start, ok = time.monotonic(), True
        if path not in built:
            print(f"Building GPU example: {path}", flush=True)
            ok = subprocess.run(["make", "-C", f"example/gpu/{path}", "-j"],
                                env=env).returncode == 0
            if ok:
                built.add(path)
        if ok:
            run_env = dict(env,
                           GPU_EXAMPLE_NAME=name,
                           GPU_EXAMPLE_PATH=path,
                           GPU_EXAMPLE_EXECUTABLE=ex["executable"],
                           GPU_EXAMPLE_VICTIM=ex["victim"],
                           GPU_EXAMPLE_EXPECTED=ex["expected"],
                           GPU_EXAMPLE_MODE=ex["mode"])
            ok = subprocess.run(
                ["bash", os.path.join(root, ".github/script/run_gpu_example.sh")],
                env=run_env).returncode == 0
        out = os.path.join(logs, name)
        os.makedirs(out, exist_ok=True)
        for log in ("server.log", "client.log", "tool.log"):
            if os.path.exists(log):
                shutil.copy(log, out)
        elapsed = time.monotonic() - start
        print("::endgroup::", flush=True)
        if not ok:
            print(f"::error title=GPU example failed::{name}", flush=True)
        results.append((name, ok, elapsed))

    failed = [name for name, ok, _ in results if not ok]
    lines = ["| GPU example | Result | Seconds |", "| --- | --- | --- |"]
    lines += [f"| {n} | {'pass' if ok else 'FAIL'} | {s:.0f} |" for n, ok, s in results]
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failed:
        print("Failed GPU examples: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
