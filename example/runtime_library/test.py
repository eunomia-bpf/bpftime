#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify the executable's lifecycle and concurrent process isolation."""
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

exe, mode = sys.argv[1:]
env = os.environ.copy()
env.update(BPFTIME_SHM_MEMORY_MB="8", BPFTIME_MAX_FD_COUNT="128")
# Prove an inherited SHM name is untouched; never use the user's default.
sentinel = Path("/dev/shm") / ("runtime-library-sentinel-" + uuid.uuid4().hex)
sentinel.write_bytes(b"owned by test, must survive")
env["BPFTIME_GLOBAL_SHM_NAME"] = sentinel.name
names = set()
children = []


def start(events, cycles):
    process = subprocess.Popen(
        [exe, str(events), str(cycles), mode], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    children.append(process)
    return process


def check(process, events, cycles):
    out, err = process.communicate(timeout=45)
    assert process.returncode == 0, (process.returncode, out, err)
    name = re.search(r"^shm=(.+)$", out, re.M).group(1)
    assert name.startswith("bpftime-runtime-library-") and name not in names
    names.add(name)
    rows = re.findall(
        r"^cycle=(\d+) events=(\d+) count=(\d+) detached_count=(\d+)$",
        out, re.M)
    assert rows == [
        tuple(map(str, (i, events, events, events)))
        for i in range(1, cycles + 1)
    ], out
    assert out.endswith("cleanup=ok\n"), out
    assert not (Path("/dev/shm") / name).exists(), name
    assert sentinel.read_bytes() == b"owned by test, must survive"


try:
    check(start(0, 3), 0, 3)
    check(start(37, 32), 37, 32)
    # Start both before waiting; each owns its counter and unique SHM.
    parallel = [start(events, 8) for events in (20000, 30000)]
    assert all(process.poll() is None for process in parallel), "no overlap"
    for process, events in zip(parallel, (20000, 30000)):
        check(process, events, 8)
    bad = subprocess.run([exe, "1", "0", mode], env=env,
                         capture_output=True, text=True, timeout=10)
    assert bad.returncode != 0 and "cycles must be positive" in bad.stderr
    # Allocation fails after the exclusive named segment has been created.
    failure_env = dict(env, BPFTIME_SHM_MEMORY_MB="1",
                       BPFTIME_MAX_FD_COUNT="1048576")
    failed = subprocess.run([exe], env=failure_env, capture_output=True,
                            text=True, timeout=10)
    assert failed.returncode != 0 and "bad_alloc" in failed.stderr, failed
    failed_name = re.search(r"^shm=(.+)$", failed.stdout, re.M).group(1)
    assert not (Path("/dev/shm") / failed_name).exists(), failed_name
    assert sentinel.read_bytes() == b"owned by test, must survive"
    print(f"{mode}: zero events, repeated lifecycles, concurrent isolation, "
          "detach, fd/SHM cleanup, failed initialization and inherited SHM "
          "preservation passed")
finally:
    for process in children:
        if process.poll() is None:
            process.kill()
            process.communicate()
    sentinel.unlink()
