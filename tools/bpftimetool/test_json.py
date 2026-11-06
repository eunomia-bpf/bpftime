#!/usr/bin/env python3
"""Exercise the JSON definition-graph contract through the public CLI."""

import copy
import json
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import sys
import tempfile


MAP = {
    "type": "bpf_map_handler", "name": "json_contract_map",
    "attr": {
        "map_type": 2, "key_size": 4, "value_size": 8, "max_entries": 1,
        "flags": 0, "ifindex": 0, "btf_vmlinux_value_type_id": 0,
        "btf_id": 0, "btf_key_type_id": 0, "btf_value_type_id": 0,
        "map_extra": 0, "kernel_bpf_map_id": 0,
    },
}
PROGRAM = {
    "type": "bpf_prog_handler", "name": "json_contract_prog",
    "attr": {"type": 1, "cnt": 2,
             "insns": "b7000000000000009500000000000000"},
}
TRACEPOINT = {
    "type": "bpf_perf_event_handler", "enabled": False,
    "attr": {"type": 2, "pid": -1, "tracepoint_id": 1,
             "data_type": "tracepoint_perf_event_data"},
}


def changed(record, **attributes):
    result = copy.deepcopy(record)
    result["attr"].update(attributes)
    return result


def test_json_contract(cli):
    shm_name = f"bpftime_json_contract_{os.getpid()}"
    shm_path = Path("/dev/shm") / shm_name
    assert not shm_path.exists(), f"refusing to reuse {shm_path}"
    env = dict(os.environ, BPFTIME_GLOBAL_SHM_NAME=shm_name,
               BPFTIME_LOG_OUTPUT="console", SPDLOG_LEVEL="err")

    def run(*args):
        return subprocess.run([str(cli), *map(str, args)], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=30)

    def success(*args):
        result = run(*args)
        assert result.returncode == 0, (args, result.returncode, result.stdout)
        return result

    def failure(*args):
        result = run(*args)
        # A C++ exception escaping a C API must not terminate the process.
        assert result.returncode > 0, (args, result.returncode, result.stdout)
        return result

    with tempfile.TemporaryDirectory(prefix="bpftime-json-contract-") as directory:
        directory = Path(directory)
        incoming = directory / "input.json"
        outgoing = directory / "output.json"

        def snapshot():
            success("export", outgoing)
            return json.loads(outgoing.read_text())

        def import_text(payload, expected=None):
            incoming.write_text(payload)
            if expected is None:
                success("import", incoming)
            else:
                failure("import", incoming)
                assert snapshot() == expected, payload

        def import_object(payload, expected=None):
            import_text(json.dumps(payload), expected)

        try:
            success("load", 7, json.dumps(MAP))
            original = snapshot()

            # Exercise the one-record API separately from the file importer.
            bad_records = [
                "", "{", "null", "[]", "0", '"handler"', "{}",
                '{"type":"bpf_map_handler"}',
                json.dumps(dict(MAP, type="unknown_handler")),
                json.dumps(dict(MAP, name=None)),
                json.dumps(dict(MAP, name="embedded\x00name")),
                json.dumps(dict(MAP, attr=[])),
                json.dumps(changed(MAP, key_size=-1)),
                json.dumps(changed(MAP, value_size=1.5)),
                json.dumps(changed(MAP, max_entries=1 << 40)),
                json.dumps(changed(MAP, value_size=65536, max_entries=65536)),
                json.dumps(changed(MAP, map_type=27, key_size=0, value_size=0,
                                   max_entries=1 << 31)),
                json.dumps(changed(MAP, flags=True)),
                json.dumps(changed(PROGRAM, cnt=-1)),
                json.dumps(changed(PROGRAM, cnt=1 << 40)),
                json.dumps(changed(PROGRAM, cnt=1)),
                json.dumps(changed(PROGRAM, insns="g7" + PROGRAM["attr"]["insns"][2:])),
                json.dumps(changed(PROGRAM, insns="7z" + PROGRAM["attr"]["insns"][2:])),
                json.dumps(changed(PROGRAM, insns="-1" + PROGRAM["attr"]["insns"][2:])),
                json.dumps(changed(PROGRAM, insns=" 1" + PROGRAM["attr"]["insns"][2:])),
                json.dumps(changed(PROGRAM, aot_insns="f")),
                json.dumps(dict(TRACEPOINT, enabled="false")),
                json.dumps(changed(TRACEPOINT, data_type="uprobe_perf_event_data")),
                json.dumps({"type": "bpf_perf_event_handler", "attr": TRACEPOINT["attr"]}),
                json.dumps({"type": "bpf_link_handler", "attr": {
                    "prog_fd": 7, "target_fd": 999, "attach_type": 41}}),
            ]
            for payload in bad_records:
                failure("load", 10, payload)
                assert snapshot() == original, payload

            # Import must validate the entire graph before publishing anything.
            for payload in ("", "{", "[]", "1", '"snapshot"', "{} {}",
                            '{"10":' + json.dumps(MAP) + ',"11":{}}'):
                import_text(payload, original)
            for key in ("-1", "7x", "07", "+7", " 7", "2147483648"):
                import_object({key: MAP}, original)
            # Duplicate keys are lost by an ordinary JSON DOM unless the
            # parser explicitly detects them before normalization.
            import_text('{"10":' + json.dumps(MAP) +
                        ',"10":' + json.dumps(MAP) + '}', original)
            import_object({"10": MAP, "11": {"type": "unknown_handler"}}, original)
            import_object({"1": MAP, "7": dict(MAP, name="must_not_replace")}, original)
            failure("load", 7, json.dumps(dict(MAP, name="must_not_replace")))
            assert snapshot() == original
            # A bad reference must not be accepted as a generic link merely
            # because an attach_type was omitted in the legacy format.
            import_object({"10": PROGRAM, "11": {
                "type": "bpf_link_handler", "attr": {
                    "prog_fd": 10, "target_fd": 7}}}, original)
            import_object({"10": PROGRAM, "11": {
                "type": "bpf_link_handler", "attr": {
                    "prog_fd": 10, "target_fd": 999}}}, original)
            failure("import", directory / "missing.json")
            failure("import", directory)
            assert snapshot() == original

            # fd order is deliberately the reverse of dependency order.
            graph = {
                "10": {"type": "bpf_link_handler", "attr": {
                    "prog_fd": 20, "target_fd": 30}},
                "20": PROGRAM, "30": TRACEPOINT,
                "21": changed(PROGRAM, aot_insns="00deadbeefff"),
                "40": {"type": "bpf_perf_event_handler", "enabled": True,
                       "attr": {"type": 1, "cpu": 0, "config": 10,
                                "sample_type": 1024,
                                "data_type": "software_perf_event_shared_ptr"}},
                "50": {"type": "bpf_perf_event_handler", "enabled": False,
                       "attr": {"type": 6, "pid": -1, "offset": (1 << 40) + 17,
                                "ref_ctr_off": (1 << 35) + 3,
                                "_module_name": "/tmp/not-attached-by-import",
                                "data_type": "uprobe_perf_event_data"}},
                "60": {"type": "bpf_perf_event_handler", "enabled": False,
                       "attr": {"type": 20000,
                                "data_type": "custom_perf_event_data",
                                "attach_argument": "custom:argument"}},
            }
            import_object(graph)
            first = snapshot()
            assert first["50"]["attr"]["offset"] == graph["50"]["attr"]["offset"]
            assert first["50"]["attr"]["ref_ctr_off"] == graph["50"]["attr"]["ref_ctr_off"]
            assert first["40"]["enabled"] is True
            assert first["21"]["attr"]["aot_insns"] == "00deadbeefff"
            assert first["60"]["attr"]["attach_argument"] == "custom:argument"
            success("remove")
            import_object(first)
            restored = snapshot()
            assert restored == first, (first, restored)
            data = directory / "input.bin"
            data.write_bytes(bytes(8))
            executed = success("run", 20, data, "type", "INTERPRET")
            assert "Return value: 0" in executed.stdout, executed.stdout

            # These failures must preserve the file and shared state. A symlink
            # must not be followed, even when it points to an ordinary file.
            sentinel = directory / "sentinel.json"
            sentinel.write_text("do not replace")
            sentinel.chmod(0o640)
            assert stat.S_IMODE(outgoing.stat().st_mode) == 0o600
            success("export", sentinel)
            assert stat.S_IMODE(sentinel.stat().st_mode) == 0o640
            sentinel.write_text("do not replace")
            alias = directory / "symlink.json"
            alias.symlink_to(sentinel)
            failure("export", alias)
            assert sentinel.read_text() == "do not replace"
            assert alias.is_symlink()
            failure("export", directory)
            failure("export", directory / "missing" / "out.json")
            fifo = directory / "output.fifo"
            os.mkfifo(fifo)
            failure("export", fifo)
            if sys.platform.startswith("linux"):
                assert stat.S_ISCHR(os.stat("/dev/full").st_mode)
                for destination in ("/dev/full", shm_path,
                                    str(shm_path.parent) + "/./" + shm_name):
                    failure("export", destination)
                    assert snapshot() == first
                # Same filesystem hard link exercises inode identity rather
                # than only a string/path comparison.
                hardlink = shm_path.with_name(shm_name + "_alias")
                try:
                    os.link(shm_path, hardlink)
                    failure("export", hardlink)
                    assert snapshot() == first
                finally:
                    hardlink.unlink(missing_ok=True)

                # Unlike /dev/full (now intentionally rejected as a device),
                # file-size limits force real write failures after opening
                # the temporary regular file, before any bytes and after a
                # short write. Ignore SIGXFSZ so the syscall returns EFBIG.
                for write_limit in (0, 16):
                    def fail_regular_file_write():
                        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
                        resource.setrlimit(resource.RLIMIT_FSIZE,
                                           (write_limit, write_limit))

                    before = sorted(p.name for p in directory.iterdir())
                    result = subprocess.run(
                        [str(cli), "export", str(sentinel)], env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, timeout=30, preexec_fn=fail_regular_file_write,
                    )
                    assert result.returncode > 0, (result.returncode, result.stdout)
                    assert sentinel.read_text() == "do not replace"
                    assert sorted(p.name for p in directory.iterdir()) == before
            assert snapshot() == first

            # Legacy attach_fds must reserve explicit IDs before synthesizing
            # links, skip standard descriptors, and not duplicate an explicit
            # link for the same edge.
            for explicit_link in (False, True):
                success("remove")
                legacy = {"0": MAP, "1": MAP, "20": changed(PROGRAM, attach_fds=[30]),
                          "30": TRACEPOINT}
                if explicit_link:
                    legacy["10"] = graph["10"]
                import_object(legacy)
                normalized = snapshot()
                link_fds = [int(fd) for fd, record in normalized.items()
                            if record["type"] == "bpf_link_handler"]
                assert link_fds == [10 if explicit_link else 3], normalized
                links = [normalized[str(fd)] for fd in link_fds]
                assert len(links) == 1, normalized
                assert links[0]["attr"]["prog_fd"] == 20
                assert links[0]["attr"]["target_fd"] == 30
                assert normalized["0"]["type"] == "bpf_map_handler"
                assert normalized["1"]["type"] == "bpf_map_handler"

            # Both the historic null encoding and {} represent an empty graph.
            success("remove")
            import_text("null")
            assert snapshot() == {}
            import_object({})
            assert snapshot() == {}
        finally:
            success("remove")
            assert not shm_path.exists(), f"leaked {shm_path}"
    print("PASS: JSON graph validation, atomic failures, and definition round-trip")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} BPFTIMETOOL")
    test_json_contract(Path(sys.argv[1]).resolve())
