#!/usr/bin/env python3
"""Exercise bpftimetool run repeat validation and real program execution."""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


# Increment the input counter and return it. The final result verifies that
# repeat controls actual executions, not just the printed configuration.
INSNS = b"".join(struct.pack("<BBhi", *insn) for insn in (
    (0x61, 0x10, 0, 0),  # r0 = *(u32 *)r1
    (0x07, 0x00, 0, 1),  # r0 += 1
    (0x63, 0x01, 0, 0),  # *(u32 *)r1 = r0
    (0x95, 0x00, 0, 0),  # exit
))
PROGRAM = {
    "type": "bpf_prog_handler",
    "name": "run_test_counter",
    "attr": {
        "type": 1, "cnt": 4, "insns": INSNS.hex(), "attach_fds": [],
    },
}


def run(cli, env, *args):
    return subprocess.run(
        [cli, *args], env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, timeout=30,
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} BPFTIMETOOL")
    cli = str(Path(sys.argv[1]).resolve())
    shm_name = f"bpftime_run_test_{os.getpid()}"
    shm_path = Path("/dev/shm") / shm_name
    assert not shm_path.exists(), f"refusing to reuse {shm_path}"
    env = dict(os.environ, BPFTIME_GLOBAL_SHM_NAME=shm_name)

    with tempfile.TemporaryDirectory(prefix="bpftimetool-run-") as directory:
        data = Path(directory) / "input.bin"
        data.write_bytes(struct.pack("<I", 0))
        try:
            result = run(cli, env, "load", "7", json.dumps(PROGRAM))
            assert result.returncode == 0, result.stdout

            for value in ("0", "-0", "-1", "typo", "", " ", "\t", "+", "-",
                          "2x", "3 ", "2147483648",
                          "99999999999999999999999999999",
                          "-99999999999999999999999999999"):
                result = run(cli, env, "run", "7", str(data), "repeat", value)
                # Require a normal diagnostic exit, never a signal/crash.
                assert result.returncode == 1, (value, result.returncode, result.stdout)
                assert "invalid repeat" in result.stdout, (value, result.stdout)
                assert "Time taken:" not in result.stdout, result.stdout

            for options in (("repeat",), ("type",), ("--repeat",), ("--type",),
                            ("repeat", "2", "type"), ("repeat", "type", "JIT"),
                            ("--repeat", "--type", "JIT")):
                result = run(cli, env, "run", "7", str(data), *options)
                assert result.returncode == 1, (options, result.returncode, result.stdout)
                assert "Usage:" in result.stdout, result.stdout
                assert "Time taken:" not in result.stdout, result.stdout

            # Duplicate options (including legacy/canonical aliases) and
            # unknown arguments are rejected uniformly.
            for options in (("repeat", "2", "repeat", "3"),
                            ("repeat", "2", "--repeat", "3"),
                            ("type", "JIT", "type", "JIT"),
                            ("unknown", "3"), ("--unknown", "3"),
                            ("repeat", "3", "unknown", "JIT"),
                            ("type", "INVALID")):
                result = run(cli, env, "run", "7", str(data), *options)
                assert result.returncode == 1, (options, result.returncode, result.stdout)
                assert "Error:" in result.stdout, result.stdout

            for options, count in (((), 1), (("repeat", "1"), 1),
                                   (("repeat", "3"), 3),
                                   (("repeat", "+3"), 3),
                                   (("repeat", " 3"), 3),
                                   (("repeat", "003"), 3),
                                   (("type", "JIT"), 1),
                                   (("type", "JIT", "repeat", "3"), 3),
                                   (("--repeat", "3", "--type", "JIT"), 3),
                                   (("--repeat=3", "type", "JIT"), 3),
                                   (("type", "JIT", "--repeat=3"), 3),
                                   (("repeat", "3", "type", "JIT"), 3)):
                result = run(cli, env, "run", "7", str(data), *options)
                assert result.returncode == 0, (options, result.returncode, result.stdout)
                assert f"Repeat N: {count} with run type JIT" in result.stdout, result.stdout
                assert f"Return value: {count}\n" in result.stdout, result.stdout
                assert "Time taken:" in result.stdout, result.stdout
            for name in ("type", "repeat"):
                named_data = Path(directory) / name
                named_data.write_bytes(struct.pack("<I", 0))
                for prefix in ((), ("--repeat=3",), ("--repeat", "3")):
                    command = [cli, "run", *prefix, "7", name]
                    result = subprocess.run(command, cwd=directory, env=env,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, timeout=30)
                    count = 3 if prefix else 1
                    assert result.returncode == 0, result.stdout
                    assert f"Return value: {count}\n" in result.stdout, result.stdout
                result = subprocess.run([cli, "run", "7", "--repeat=3", name],
                                        cwd=directory, env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, timeout=30)
                assert result.returncode == 0 and "Return value: 3\n" in result.stdout, result.stdout
            result = run(cli, env, "run", "7", str(data), "repeat", "3", "type", "INTERPRET")
            assert result.returncode == 0 and "Return value: 3\n" in result.stdout, result.stdout
            result = run(cli, env, "run", "7", str(data), "type", "AOT")
            assert result.returncode == 1 and "AOT instructions not found" in result.stdout, result.stdout
        finally:
            result = run(cli, env, "remove")
            assert result.returncode == 0, result.stdout
            assert not shm_path.exists(), f"leaked {shm_path}"
    print("PASS: bpftimetool run repeat validation and execution")


if __name__ == "__main__":
    main()
