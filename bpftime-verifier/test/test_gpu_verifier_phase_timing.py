#!/usr/bin/env python3
"""CPU-only subprocess tests for opt-in GPU-verifier phase diagnostics."""

import json
import os
import subprocess
import sys


PREFIX = "BPFTIME_GPU_VERIFIER_PHASE_TIMING "
ENVIRONMENT = "BPFTIME_GPU_VERIFIER_PHASE_TIMING"


def run_tests(binary: str, setting: str | None) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if setting is None:
        environment.pop(ENVIRONMENT, None)
    else:
        environment[ENVIRONMENT] = setting
    return subprocess.run(
        [binary, "[gpu]", "--reporter", "compact"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def require_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise AssertionError(
            f"{label} verifier tests failed ({result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def phase_records(stderr: str) -> list[dict[str, object]]:
    records = []
    for line in stderr.splitlines():
        if line.startswith(PREFIX):
            records.append(json.loads(line[len(PREFIX) :]))
    return records


def duration_mask(record: dict[str, object]) -> int:
    wall = record["wall_ns"]
    assert isinstance(wall, dict)
    mask = 0
    for bit, phase in enumerate(
        ("input_copy", "validation", "prevail", "uniformity", "simt")
    ):
        value = wall[phase]
        if value is not None:
            assert isinstance(value, int) and value >= 0
            mask |= 1 << bit
    assert isinstance(wall["total"], int) and wall["total"] >= 0
    return mask


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_gpu_verifier_phase_timing.py TEST_BINARY")
    binary = sys.argv[1]

    for setting, label in ((None, "unset"), ("0", "zero")):
        result = run_tests(binary, setting)
        require_success(result, label)
        if PREFIX in result.stderr:
            raise AssertionError(f"{label} unexpectedly enabled phase timing")

    enabled = run_tests(binary, "1")
    require_success(enabled, "enabled")
    records = phase_records(enabled.stderr)
    if not records:
        raise AssertionError("enabled run emitted no phase records")

    for record in records:
        assert record["schema"] == "bpftime-gpu-verifier-phase-timing-v1"
        assert isinstance(record["instruction_count"], int)
        assert isinstance(record["map_count"], int)
        assert isinstance(record["accepted"], bool)
        assert record["phase_mask"] == duration_mask(record)
        process_cpu = record["process_cpu_ns"]
        assert isinstance(process_cpu, dict)
        for phase in (
            "input_copy",
            "validation",
            "prevail",
            "uniformity",
            "simt",
            "total",
        ):
            value = process_cpu[phase]
            assert value is None or isinstance(value, int) and value >= 0

    assert any(record["accepted"] and record["phase_mask"] == 31 for record in records)
    assert any(
        not record["accepted"] and record["phase_mask"] == 7 for record in records
    )
    assert any(
        not record["accepted"] and record["phase_mask"] == 31 for record in records
    )
    assert any(
        not record["accepted"]
        and record["instruction_count"] == 1
        and record["phase_mask"] == 2
        for record in records
    )
    assert any(
        not record["accepted"]
        and record["instruction_count"] == 0
        and record["phase_mask"] == 3
        for record in records
    )
    assert any(
        not record["accepted"]
        and record["map_count"] == 1
        and record["phase_mask"] == 3
        for record in records
    )
    assert any(
        not record["accepted"] and record["phase_mask"] == 0 for record in records
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
