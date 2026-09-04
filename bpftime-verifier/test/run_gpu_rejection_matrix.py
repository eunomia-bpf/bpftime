#!/usr/bin/env python3
"""Run bpftime's CPU-only verifier pairs; never load a kernel or use a GPU."""

import argparse
from pathlib import Path
import subprocess
import sys


CASES = (
    ("gpu-prevail", "memory-bounds", "revision base verifier bounds pair"),
    ("gpu-prevail", "termination", "revision base verifier loop pair"),
    ("gpu-simt", "warp-uniform-branch", "revision SIMT branch pair"),
    ("gpu-simt", "shared-map-side-effects", "revision SIMT map side-effect pairs"),
    ("gpu-simt", "non-uniform-atomic", "revision SIMT atomic pair"),
    (
        "gpu-simt",
        "no-global-synchronization",
        "revision SIMT global synchronization helper pair",
    ),
)


def run_case(binary: Path, layer: str, policy: str, test_name: str) -> bool:
    result = subprocess.run(
        [str(binary), test_name, "--rng-seed", "1"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        print(
            f'FAIL layer={layer} policy={policy} test="{test_name}"',
            file=sys.stderr,
        )
        print(result.stdout, file=sys.stderr, end="")
        return False
    print(
        f"PASS layer={layer} policy={policy} unsafe=rejected "
        f'control=accepted test="{test_name}"'
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CPU-only PREVAIL and GPU SIMT rejection/control pairs"
    )
    parser.add_argument("verifier_tests", type=Path)
    args = parser.parse_args()

    binary = args.verifier_tests.resolve()
    if not binary.is_file():
        parser.error(f"verifier test binary is absent: {binary}")

    passed = sum(run_case(binary, *case) for case in CASES)
    print(
        "NOT_RUN layer=host-linux-verifier policy=termination,memory,kfunc "
        'reason="requires separate kernel-load fixtures"'
    )
    print(
        "NOT_RUN layer=driver-transition-validator policy=invalid-transition "
        'reason="driver production-header test is outside bpftime"'
    )
    print(
        f"SUMMARY cpu_only=1 device_execution=0 passed={passed} "
        f"required={len(CASES)} external_not_run=2"
    )
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
