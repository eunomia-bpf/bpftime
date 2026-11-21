#!/usr/bin/env python3
"""Exercise bpftimetool load arguments and JSON import/export."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from test_json import test_json_contract


MAP = {
    "type": "bpf_map_handler",
    "name": "load_test_map",
    "attr": {
        "map_type": 2,
        "key_size": 4,
        "value_size": 8,
        "max_entries": 1,
        "flags": 0,
        "ifindex": 0,
        "btf_vmlinux_value_type_id": 0,
        "btf_id": 0,
        "btf_key_type_id": 0,
        "btf_value_type_id": 0,
        "map_extra": 0,
        "kernel_bpf_map_id": 0,
    },
}


def run(cli, env, *args, cwd=None):
    return subprocess.run(
        [cli, *args], env=env, cwd=cwd, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, timeout=30,
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} BPFTIMETOOL")
    cli = str(Path(sys.argv[1]).resolve())
    shm_name = f"bpftime_load_test_{os.getpid()}"
    shm_path = Path("/dev/shm") / shm_name
    assert not shm_path.exists(), f"refusing to reuse {shm_path}"
    env = dict(os.environ, BPFTIME_GLOBAL_SHM_NAME=shm_name,
               BPFTIME_LOG_OUTPUT="console", SPDLOG_LEVEL="err")
    payload = json.dumps(MAP)

    with tempfile.TemporaryDirectory(prefix="bpftimetool-load-") as directory:
        exported = Path(directory) / "state.json"
        try:
            for args in (("load", "7"),
                         ("load", "7", payload, "extra")):
                result = run(cli, env, *args)
                assert result.returncode == 1, result.stdout
                assert "Usage:" in result.stdout, result.stdout
                assert not shm_path.exists(), f"created {shm_path}"

            # A malformed fd must be rejected instead of being silently
            # converted to fd 0, which would clobber unrelated state.
            for bad_fd in ("typo", "-1", "", "7x", "99999999999999999999"):
                result = run(cli, env, "load", bad_fd, payload)
                assert result.returncode == 1, (bad_fd, result.stdout)
                assert not shm_path.exists(), f"created {shm_path}"

            result = run(cli, env, "load", "7", payload)
            assert result.returncode == 0, result.stdout

            # An fd outside the handler table must fail, not report success.
            # Operation failures return the raw negative result, as the other
            # commands in this tool do. Require normal failure, never a signal.
            result = run(cli, env, "load", "6144", payload)
            assert result.returncode > 0, (result.returncode, result.stdout)

            result = run(cli, env, "export", str(exported))
            assert result.returncode == 0, result.stdout
            state = json.loads(exported.read_text())
            assert sorted(state) == ["7"], sorted(state)
            handler = state["7"]
            assert handler["type"] == "bpf_map_handler", handler
            assert handler["name"] == "load_test_map", handler

            # Opening an output directory or a missing parent must fail.
            for destination in (Path(directory),
                                Path(directory) / "missing" / "state.json"):
                result = run(cli, env, "export", str(destination))
                assert result.returncode > 0, (destination, result.stdout)
                assert str(destination) in result.stdout, result.stdout

            # Export accepts regular files only. Reject /dev/full for both
            # small and large graphs without modifying shared state. The
            # contract suite separately injects a regular-file write error.
            if sys.platform.startswith("linux"):
                for large in (False, True):
                    if large:
                        large_map = dict(MAP, name="export_test_" + "x" * 65536)
                        result = run(cli, env, "load", "8", json.dumps(large_map))
                        assert result.returncode == 0, result.stdout
                    result = run(cli, env, "export", "/dev/full")
                    assert result.returncode > 0, result.stdout
                    assert "/dev/full" in result.stdout, result.stdout

                # A failed export must not damage shared state or prevent a
                # subsequent successful export, including the large payload.
                result = run(cli, env, "export", str(exported))
                assert result.returncode == 0, result.stdout
                state = json.loads(exported.read_text())
                assert sorted(state) == ["7", "8"], sorted(state)
                assert state["7"] == handler, state["7"]
                assert state["8"]["name"] == large_map["name"], state["8"]
            result = run(cli, env, "remove")
            assert result.returncode == 0 and not shm_path.exists(), result.stdout
            result = run(cli, env, "import", str(exported))
            assert result.returncode == 0, result.stdout
            restored = Path(directory) / "restored.json"
            result = run(cli, env, "export", str(restored))
            assert result.returncode == 0, result.stdout
            assert json.loads(restored.read_text()) == state
        finally:
            result = run(cli, env, "remove")
            assert result.returncode == 0 and not shm_path.exists(), result.stdout
    test_json_contract(cli)
    print("PASS: bpftimetool load arguments and JSON import/export")


if __name__ == "__main__":
    main()
