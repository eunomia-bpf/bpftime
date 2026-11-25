#!/usr/bin/env python3
"""Check command schemas and CUDA parsing without requiring a CUDA runtime."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from test_load import MAP, run


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {sys.argv[0]} BPFTIMETOOL CXX")
    cli = str(Path(sys.argv[1]).resolve())
    shm_name = f"bpftime_arguments_test_{os.getpid()}"
    shm_path = Path("/dev/shm") / shm_name
    assert not shm_path.exists(), shm_path
    env = dict(os.environ, BPFTIME_GLOBAL_SHM_NAME=shm_name)
    failures = [
        (), ("unknown",), ("--unknown",), ("remove", "extra"),
        ("load",), ("load", "7"), ("load", "7", "{}", "extra"),
        ("import",), ("import", "a", "extra"),
        ("export",), ("export", "a", "extra"),
        ("run",), ("run", "7"), ("run", "7", "input", "extra"),
        ("run-on-cuda", "program"),  # Disabled in this CPU tool build.
    ]
    for command in ("load", "run"):
        for value in ("", " ", "+", "-", "-1", "1x", "1 ",
                      "2147483648", "99999999999999999999999"):
            failures.append((command, value, "{}" if command == "load" else "input"))
    with tempfile.TemporaryDirectory(prefix="bpftimetool-arguments-") as directory:
        directory = Path(directory)
        try:
            for args in failures:
                result = run(cli, env, *args)
                assert result.returncode == 1, (args, result.returncode, result.stdout)
                assert "Error:" in result.stdout and "Usage:" in result.stdout, result.stdout
                assert not shm_path.exists(), f"created state for invalid args: {args}"

            for command in ((), ("load",), ("import",), ("export",), ("remove",), ("run",)):
                result = run(cli, env, *command, "--help")
                assert result.returncode == 0 and "Usage:" in result.stdout, result.stdout
                assert not shm_path.exists(), shm_path

            result = run(cli, env, "load", "0", json.dumps(MAP))
            assert result.returncode == 0, result.stdout
            # Extra arguments must not turn a rejected remove into a deletion.
            result = run(cli, env, "remove", "extra")
            assert result.returncode == 1 and shm_path.exists(), result.stdout
            result = run(cli, env, "remove", "--help")
            assert result.returncode == 0 and shm_path.exists(), result.stdout
            exported = directory / "state.json"
            result = run(cli, env, "export", str(exported))
            assert result.returncode == 0, result.stdout
            assert json.loads(exported.read_text())["0"]["name"] == MAP["name"]
            result = run(cli, env, "export", "./--help", cwd=directory)
            assert result.returncode == 0, result.stdout
            assert json.loads((directory / "--help").read_text())["0"]["name"] == MAP["name"]
        finally:
            result = run(cli, env, "remove")
            assert result.returncode == 0 and not shm_path.exists(), result.stdout

        # Compile the actual shared schema; do not duplicate parser logic or
        # require CUDA headers/libraries just to verify its argument contract.
        source = directory / "cuda_parser.cpp"
        source.write_text(r"""
#include "arguments.hpp"
#include <iostream>
int main(int argc, char **argv) {
    tool_arguments args(argv[0], true);
    try {
        args.parse(argc, argv);
        auto dims = args.cuda.get<std::vector<int>>("dimensions");
        std::cout << args.cuda.get<std::string>("program") << " "
                  << args.cuda.present<int>("count").value_or(1);
        for (int dim : dims) std::cout << " " << dim;
        std::cout << "\n";
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
""")
        repo = Path(__file__).resolve().parents[2]
        parser = directory / "cuda_parser"
        subprocess.run([sys.argv[2], "-std=c++17", str(source), "-o", str(parser),
                        "-I", str(repo / "tools/bpftimetool"),
                        "-I", str(repo / "third_party/argparse/include")],
                       check=True, timeout=120)
        for tail, expected in [
            ((), "program 1"), (("3",), "program 3"),
            (("3", "1", "2", "3", "4", "5", "6"), "program 3 1 2 3 4 5 6"),
        ]:
            result = run(str(parser), env, "run-on-cuda", "program", *tail)
            assert result.returncode == 0 and result.stdout.strip() == expected, result.stdout
        invalid_cuda = [("0",), ("-1",), ("typo",), ("2147483648",)]
        invalid_cuda += [("3",) + ("1",) * n for n in (1, 2, 3, 4, 5, 7)]
        for index in range(6):
            for value in ("0", "-1", "x", "2147483648"):
                dims = ["1"] * 6
                dims[index] = value
                invalid_cuda.append(("3", *dims))
        for tail in invalid_cuda:
            result = run(str(parser), env, "run-on-cuda", "program", *tail)
            assert result.returncode == 1 and "Error:" in result.stdout, result.stdout
    print("PASS: command errors, help, state isolation and CUDA argument schema")


if __name__ == "__main__":
    main()
