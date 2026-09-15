#!/usr/bin/env python3
"""Exercise bpftimetool load argument handling and JSON import."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


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


def run(cli, env, *args):
    return subprocess.run(
        [cli, *args], env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, timeout=30,
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} BPFTIMETOOL")
    cli = str(Path(sys.argv[1]).resolve())
    shm_name = f"bpftime_load_test_{os.getpid()}"
    shm_path = Path("/dev/shm") / shm_name
    env = dict(os.environ, BPFTIME_GLOBAL_SHM_NAME=shm_name)
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

            result = run(cli, env, "load", "7", payload)
            assert result.returncode == 0, result.stdout
            result = run(cli, env, "export", str(exported))
            assert result.returncode == 0, result.stdout
            handler = json.loads(exported.read_text())["7"]
            assert handler["type"] == "bpf_map_handler", handler
            assert handler["name"] == "load_test_map", handler
        finally:
            run(cli, env, "remove")
    print("PASS: bpftimetool load argument handling")


if __name__ == "__main__":
    main()
