#!/usr/bin/env python3
"""Build and run every example of one CI configuration in one job.

The examples workflow used to start one job per example and configuration,
about 144 per push, and each spent about two minutes on checkout, container
start, downloads and coverage around about 15 seconds of test. This runs
the examples one after another in the job of their configuration, reports
each in its own log group and in the job summary, and fails the job when
any example fails.

Environment: PRIVILEGED ("true" in the privileged configuration); the
prebuilt runtime is in ~/.bpftime and BPFTIME_VM_NAME is already set.
"""
import glob
import os
import subprocess
import sys
import time

# The examples and expected output of the former matrix (which listed
# opensnoop twice under two names; it runs once here). Examples that
# trace syscalls, and the tail-call and malloc examples, need the privileged
# container (tracefs, BPF), as before.
EXAMPLES = [
    dict(name="statsnoop", path="tracing/statsnoop", executable="./statsnoop", victim="./victim", syscall_trace=True, expected="victim               0    0    /sys"),
    dict(name="malloc", path="malloc", executable="./malloc", victim="./victim", syscall_trace=False, expected="malloc calls: ", privileged=True),
    dict(name="minimal", path="minimal", executable="./uprobe", victim="./victim", syscall_trace=False, expected=""),
    dict(name="opensnoop", path="tracing/opensnoop", executable="./opensnoop", victim="./victim", syscall_trace=True, expected="   0 test.txt"),
    dict(name="sslsniff", path="tracing/sslsniff", executable="./sslsniff", victim="/bin/wget https://www.google.com", syscall_trace=False, expected="----- DATA -----"),
    dict(name="bashreadline", path="tracing/bashreadline", executable="./readline", victim="/bin/bash", syscall_trace=False, expected="info"),
    dict(name="syscount", path="tracing/syscount", executable="./syscount", victim="/bin/bash", syscall_trace=False, expected="info"),
    dict(name="funclatency", path="tracing/funclatency", executable="./funclatency -i 1 ./victim:plus", victim="./victim", syscall_trace=False, expected="|*"),
    dict(name="mountsnoop", path="tracing/mountsnoop", executable="./mountsnoop", victim="./victim", syscall_trace=True, expected="mount("),
    dict(name="sigsnoop", path="tracing/sigsnoop", executable="./sigsnoop", victim="./victim", syscall_trace=True, expected="victim           0         -1      0"),
    dict(name="tailcall_minimal", path="bpf_features/tailcall_minimal", executable="./tailcall_minimal", victim="./victim", syscall_trace=False, expected="See /sys/kernel/debug/tracing/trace_pipe for output (5)", privileged=True),
    dict(name="usdt_minimal", path="minimal/usdt_minimal", executable="./usdt_minimal", victim="./victim", syscall_trace=False, expected="bpf:"),
    dict(name="get_stack_id_example", path="bpf_features/get_stack_id_example", executable="./get_stack_id_example", victim="./victim", syscall_trace=False, expected="Function: malloc enter"),
    dict(name="bloom_filter_demo", path="bpf_features/bloom_filter_demo", executable="./uprobe_bloom_filter", victim="./target", syscall_trace=False, expected="Bloom Filter Real-time Monitoring Statistics"),
    dict(name="lpm_trie_demo", path="bpf_features/lpm_trie_demo", executable="./file_access_monitor", victim="./file_access_target", syscall_trace=False, expected="All tests passed! bpftime LPM Trie working correctly"),
    dict(name="queue_demo", path="bpf_features/queue_demo", executable="./uprobe_queue", victim="./target", syscall_trace=False, expected="Queue status: non-empty"),
    dict(name="stack_demo", path="bpf_features/queue_demo", executable="./uprobe_stack", victim="./target", syscall_trace=False, expected="Starting to pop events from stack"),
]

INCLUDE = [
    "build/libbpf", "build/libbpf/bpf",
    "third_party/bpftool/build/usr/include", "third_party/bpftool/build/usr/include/bpf",
    "third_party/bpftool/build/include", "third_party/bpftool/build/include/bpf",
]


def main() -> int:
    root = os.getcwd()
    privileged = os.environ.get("PRIVILEGED") == "true"
    home = os.path.expanduser("~")
    cli = f"{home}/.bpftime/bpftime -i {home}/.bpftime"
    env = dict(os.environ)
    for var in ("C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        env[var] = ":".join(INCLUDE + ([env[var]] if env.get(var) else []))

    built, results = set(), []
    for ex in EXAMPLES:
        if ex.get("privileged") and not privileged:
            continue
        if ex["syscall_trace"] and not privileged:
            continue
        name, path = ex["name"], ex["path"]
        print(f"::group::{name}", flush=True)
        start, ok = time.monotonic(), True
        if path not in built:
            if path == "bpf_features/lpm_trie_demo":
                # clang rejects this flag for the BPF target here.
                makefile = os.path.join(root, "example", path, "Makefile")
                with open(makefile) as f:
                    text = f.read()
                with open(makefile, "w") as f:
                    f.write(text.replace("-mllvm -bpf-stack-size=8192", ""))
            ok = subprocess.run(["make", "-C", f"example/{path}", "-j"],
                                env=env).returncode == 0
            if ok:
                built.add(path)
        if ok:
            # Each example starts from no bpftime shared memory. Two runs
            # may race on the same segment, so a file that is already gone
            # is not a failure.
            for leftover in glob.glob("/dev/shm/bpftime*"):
                try:
                    os.remove(leftover)
                except OSError:
                    pass
            ok = subprocess.run(
                ["timeout", "-k", "30s", "10m", "python3",
                 os.path.join(root, ".github/script/run_example.py"),
                 ex["executable"], ex["victim"], ex["expected"], cli,
                 "1" if ex["syscall_trace"] else "0"],
                cwd=os.path.join(root, "example", path), env=env).returncode == 0
        elapsed = time.monotonic() - start
        print("::endgroup::", flush=True)
        if not ok:
            print(f"::error title=Example failed::{name}", flush=True)
        results.append((name, ok, elapsed))

    failed = [name for name, ok, _ in results if not ok]
    lines = ["| Example | Result | Seconds |", "| --- | --- | --- |"]
    lines += [f"| {n} | {'pass' if ok else 'FAIL'} | {s:.0f} |" for n, ok, s in results]
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failed:
        print("Failed examples: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
