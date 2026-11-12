#!/usr/bin/env python3
"""Exercise bpftime-aot run with native objects and real input files."""

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


FIXTURE = """
#include <stddef.h>
#include <stdint.h>

uint64_t bpf_main(void *memory, size_t size)
{
#ifdef RETURN_NEGATIVE
    return UINT64_MAX;
#else
    const uint8_t *bytes = memory;
    if (size == 0)
        return 0;
    return ((uint64_t)size << 32) | bytes[0] |
           ((uint64_t)bytes[size - 1] << 8);
#endif
}
"""


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {sys.argv[0]} BPFTIME_AOT CC")
    cli = str(Path(sys.argv[1]).resolve())
    compiler = sys.argv[2]
    env = dict(os.environ, BPFTIME_VM_NAME="llvm", SPDLOG_LEVEL="info")
    failures = 0
    with tempfile.TemporaryDirectory(prefix="bpftime-aot-run-") as directory:
        work = Path(directory)
        source = work / "fixture.c"
        source.write_text(FIXTURE)
        for name, flags in (
            ("program", []),
            ("negative", ["-DRETURN_NEGATIVE"]),
            ("missing-entry", ["-Dbpf_main=wrong_entry"]),
        ):
            subprocess.run(
                [compiler, "-O2", "-fPIC", "-c", str(source),
                 "-o", str(work / f"{name}.o"), *flags],
                check=True, timeout=30,
            )
        (work / "short.bin").write_bytes(bytes([42, 0, 0, 0]))
        (work / "long.bin").write_bytes(bytes([17] + [0] * 62 + [34]))
        (work / "empty.bin").write_bytes(b"")
        (work / "malformed.o").write_bytes(b"not an ELF object\n")

        cases = (
            ("four-byte input", "program.o", "short.bin", 0, 17179869226),
            ("64-byte input", "program.o", "long.bin", 0, 274877915665),
            ("omitted input", "program.o", None, 0, 0),
            ("empty input", "program.o", "empty.bin", 0, 0),
            ("negative program result", "negative.o", None, 0,
             18446744073709551615),
            ("malformed object", "malformed.o", None, 1, None),
            ("missing entry point", "missing-entry.o", None, 1, None),
            ("missing object", "absent.o", None, 1, None),
            ("missing input", "program.o", "absent.bin", 1, None),
        )
        for name, obj, memory, status, value in cases:
            args = [cli, "run", str(work / obj)]
            if memory is not None:
                args.append(str(work / memory))
            result = subprocess.run(
                args, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, timeout=30,
            )
            outputs = re.findall(r"\bOutput: (\d+)\b", result.stdout)
            expected = [] if value is None else [str(value)]
            if result.returncode != status or outputs != expected:
                failures += 1
                print(f"FAIL: {name}: expected exit={status}, output={expected}; "
                      f"got exit={result.returncode}, output={outputs}",
                      file=sys.stderr)
                print(result.stdout, file=sys.stderr)
            else:
                print(f"PASS: {name}")
    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
