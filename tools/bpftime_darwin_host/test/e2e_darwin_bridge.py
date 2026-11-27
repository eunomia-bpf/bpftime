#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Native macOS end-to-end test for bpftime-darwin-host.

This standalone PR has no DYLD preload agent yet, so the test drives every
host daemon RPC directly: hello, synth_elf for Mach-O and dyld_shared_cache
libSystem images, .eh_frame / .note.stapsdt-capable ELF struct validation,
map CRUD across every required map type, prog_load LDDW relocation,
spawn_cmd cold-start hold via POSIX_SPAWN_START_SUSPENDED + SIGCONT, and a
bounded 64-event ringbuf / software-perf stream produced through the public
runtime SHM APIs.
"""
import json
import os
import socket
import struct
import subprocess
import sys
import time

BPF_MAP_HASH = 1
BPF_MAP_ARRAY = 2
BPF_MAP_PROG_ARRAY = 3
BPF_MAP_PERF_EVENT_ARRAY = 4
BPF_MAP_PERCPU_HASH = 5
BPF_MAP_PERCPU_ARRAY = 6
BPF_MAP_STACK_TRACE = 7
BPF_MAP_LPM_TRIE = 11
BPF_MAP_RINGBUF = 27

EVENT_COUNT = 64
PAYLOAD_SIZE = 32


def send_frame(sock_, json_obj, blob=b""):
    payload = json.dumps(json_obj).encode()
    sock_.sendall(struct.pack("<II", len(payload), len(blob)) + payload + blob)


def recv_exact(sock_, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock_.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed during recv")
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(sock_):
    json_len, blob_len = struct.unpack("<II", recv_exact(sock_, 8))
    payload = recv_exact(sock_, json_len) if json_len else b""
    blob = recv_exact(sock_, blob_len) if blob_len else b""
    return json.loads(payload.decode()), blob


class ElfReader:
    """Minimal little-endian ELF64 parser (no third-party dependency)."""

    def __init__(self, data):
        self.data = data
        assert data[:4] == b"\x7fELF", "not an ELF file"
        assert data[4] == 2, "not ELF64"
        (self.e_type, self.e_machine, self.e_version, self.e_entry,
         self.e_phoff, self.e_shoff, self.e_flags, self.e_ehsize,
         self.e_phentsize, self.e_phnum, self.e_shentsize, self.e_shnum,
         self.e_shstrndx) = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
        self.sections = [
            struct.unpack_from("<IIQQQQIIQQ", data,
                               self.e_shoff + i * self.e_shentsize)
            for i in range(self.e_shnum)
        ]
        shstr = self.sections[self.e_shstrndx]
        self.shstrtab = data[shstr[4]:shstr[4] + shstr[5]]

    def section_name(self, sh):
        end = self.shstrtab.index(b"\x00", sh[0])
        return self.shstrtab[sh[0]:end].decode()

    def find_section(self, name):
        return next((sh for sh in self.sections
                     if self.section_name(sh) == name), None)

    def symbols(self, sh):
        strtab_sh = self.sections[sh[6]]
        strtab = self.data[strtab_sh[4]:strtab_sh[4] + strtab_sh[5]]
        out = {}
        for off in range(sh[4], sh[4] + sh[5], sh[9]):
            st_name, st_info, st_other, st_shndx, st_value, st_size = \
                struct.unpack_from("<IBBHQQ", self.data, off)
            if st_name == 0:
                continue
            end = strtab.index(b"\x00", st_name)
            out[strtab[st_name:end].decode()] = (st_value, st_size)
        return out

    def program_headers(self):
        return [
            struct.unpack_from("<IIQQQQQQ", self.data,
                               self.e_phoff + i * self.e_phentsize)
            for i in range(self.e_phnum)
        ]


class BridgeClient:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=30)

    def call(self, obj, blob=b""):
        send_frame(self.s, obj, blob)
        resp, rblob = recv_frame(self.s)
        if not resp.get("ok"):
            raise AssertionError(f"op {obj.get('op')} failed: {resp}")
        return resp, rblob

    def close(self):
        self.s.close()


def run_tests(port, victim, producer):
    cli = BridgeClient(port)

    # 1. hello
    hello, _ = cli.call({"op": "hello"})
    assert hello["arch"] == "arm64" and hello["num_cpus"] >= 1, hello
    shm_name = hello["shm"]
    print(f"[1] hello ok: arch={hello['arch']} cpus={hello['num_cpus']} "
          f"shm={shm_name}")

    # 2. synth_elf: victim Mach-O stub
    _, victim_elf_bytes = cli.call({"op": "synth_elf", "path": victim})
    victim_elf = ElfReader(victim_elf_bytes)
    assert victim_elf.e_type == 3 and victim_elf.e_machine == 183
    load_segs = [ph for ph in victim_elf.program_headers() if ph[0] == 1]
    assert load_segs and load_segs[0][2] == 0 and load_segs[0][3] == 0, \
        "PT_LOAD must be p_offset=0,p_vaddr=0"
    assert victim_elf.find_section(".dynsym") and \
        victim_elf.find_section(".symtab"), "both symbol tables required"
    assert victim_elf.find_section(".eh_frame"), ".eh_frame required"
    syms = victim_elf.symbols(victim_elf.find_section(".symtab"))
    dynsyms = victim_elf.symbols(victim_elf.find_section(".dynsym"))
    assert "compute" in syms and "compute" in dynsyms
    compute_off, compute_size = syms["compute"]
    assert compute_size > 0, "symbols need positive st_size for uprobe checks"
    print(f"[2a] victim stub ELF ok: compute=0x{compute_off:x} "
          f"size={compute_size}, sections="
          f"{[victim_elf.section_name(s) for s in victim_elf.sections[1:]]}")

    # 3. synth_elf: dyld_shared_cache libSystem umbrella
    libsys = "/bpftime-dyld/usr/lib/libSystem.B.dylib"
    _, sys_elf_bytes = cli.call({"op": "synth_elf", "path": libsys})
    sys_elf = ElfReader(sys_elf_bytes)
    assert sys_elf.e_type == 3 and sys_elf.e_machine == 183
    sys_syms = sys_elf.symbols(sys_elf.find_section(".symtab"))
    assert "malloc" in sys_syms and "free" in sys_syms
    malloc_synth_off = sys_syms["malloc"][0]
    assert malloc_synth_off >= 0x10000000, "expected synthetic offset"
    print(f"[2b] libSystem stub ok: malloc=0x{malloc_synth_off:x}, "
          f"{len(sys_syms)} aggregated exports")

    # 4. resolve_lib aliases
    for name in ("libc", "libc.so.6", "pthread"):
        r, _ = cli.call({"op": "resolve_lib", "name": name})
        assert r["path"] == libsys, r
    r, _ = cli.call({"op": "resolve_lib", "name": "libfoo"})
    assert r["path"] == "/bpftime-dyld/usr/lib/libfoo.dylib"
    print("[3] resolve_lib aliases ok")

    # 5. proc_maps virtualization (agent-published snapshot format)
    maps_path = f"/tmp/bpftime-maps-{os.getpid()}.txt"
    with open(maps_path, "w") as f:
        f.write("100000000-100100000 r-xp 0 0 0 " + victim + "\n")
        f.write("180000000-180200000 r-xp 0 0 0 /usr/lib/libSystem.B.dylib\n")
    pm, maps_blob = cli.call({"op": "proc_maps", "pid": os.getpid()})
    os.unlink(maps_path)
    maps_text = maps_blob.decode()
    assert victim in maps_text and "/bpftime-dyld/usr/lib/" in maps_text, \
        maps_text
    print("[4] proc_maps dyld path virtualization ok")

    # 6. map_create across every required type
    def make_map(map_type, key_size, value_size, max_entries, name):
        resp, _ = cli.call({
            "op": "map_create", "map_type": map_type,
            "key_size": key_size, "value_size": value_size,
            "max_entries": max_entries, "flags": 0, "name": name})
        return resp["host_map_id"]
    ring = make_map(BPF_MAP_RINGBUF, 0, 0, 65536, "rb")
    make_map(BPF_MAP_ARRAY, 4, 8, 1, "array1")
    make_map(BPF_MAP_LPM_TRIE, 12, 8, 16, "lpm")
    make_map(BPF_MAP_PERF_EVENT_ARRAY, 4, 4, 16, "pea")
    percpu = make_map(BPF_MAP_PERCPU_ARRAY, 4, 8, 4, "percpu")
    hmap = make_map(BPF_MAP_HASH, 8, 8, 64, "hash")
    # PROG_ARRAY / STACK_TRACE creation is compiled only with
    # BPFTIME_BUILD_WITH_LIBBPF (off on Darwin); the Darwin preload agent
    # PR enables their pure-userspace paths separately.
    ncpu = hello["num_cpus"]
    cli.call({"op": "map_update", "host_map_id": percpu, "flags": 0,
              "is_percpu": True},
             struct.pack("<I", 0) + b"".join(
                 struct.pack("<Q", 1000 + c) for c in range(ncpu)))
    pl, pb = cli.call({"op": "map_lookup", "host_map_id": percpu},
                      struct.pack("<I", 0))
    assert pl["is_percpu"] and pl["num_cpus"] == ncpu
    assert struct.unpack(f"<{ncpu}Q", pb[:ncpu * 8]) == \
        tuple(range(1000, 1000 + ncpu))
    print(f"[5] map types created; per-cpu round-trip ok ({ncpu} cpus)")

    # 7. hash CRUD: update, lookup, get_next_key, delete, batch
    key1, val1 = struct.pack("<Q", 1), struct.pack("<Q", 11)
    key2, val2 = struct.pack("<Q", 2), struct.pack("<Q", 22)
    cli.call({"op": "map_update", "host_map_id": hmap}, key1 + val1)
    cli.call({"op": "map_update", "host_map_id": hmap}, key2 + val2)
    _, vb = cli.call({"op": "map_lookup", "host_map_id": hmap}, key1)
    assert struct.unpack("<Q", vb)[0] == 11
    _, nk = cli.call({"op": "map_get_next_key", "host_map_id": hmap,
                      "has_key": False})
    assert nk in (key1, key2)
    _, nk2 = cli.call({"op": "map_get_next_key", "host_map_id": hmap,
                       "has_key": True}, nk)
    if nk == key1:
        assert nk2 == key2
    end, _ = cli.call({"op": "map_get_next_key", "host_map_id": hmap,
                       "has_key": True}, key2)
    assert end.get("end") is True
    cli.call({"op": "map_delete", "host_map_id": hmap}, key1)
    err = None
    try:
        cli.call({"op": "map_lookup", "host_map_id": hmap}, key1)
    except AssertionError as ex:
        err = ex
    assert err is not None, "deleted key still present"
    cli.call({"op": "map_update_batch", "host_map_id": hmap, "count": 1},
             key1 + val1)
    _, vb = cli.call({"op": "map_lookup", "host_map_id": hmap}, key1)
    assert struct.unpack("<Q", vb)[0] == 11
    print("[6] hash map update/lookup/iterate/delete/batch ok")

    # 8. prog_load with LDDW fd relocation
    exit_insn = struct.pack("<BBhi", 0x95, 0, 0, 0)
    # byte 0x11: dst=r1, src=1 (BPF_PSEUDO_MAP_FD) triggers relocation
    lddw = struct.pack("<BBhi", 0x18, 0x11, 0, 9001) + b"\x00" * 8
    fd_map = [{"vm_fd": 9001, "host_map_id": ring}]
    pid_prog, _ = cli.call(
        {"op": "prog_load", "prog_type": 0, "name": "p1",
         "license": "MIT", "fd_map": fd_map}, lddw + exit_insn)
    assert pid_prog["host_prog_id"] >= 0
    bad = None
    try:
        cli.call({"op": "prog_load", "prog_type": 0, "name": "p2",
                  "fd_map": []}, lddw + exit_insn)
    except AssertionError as ex:
        bad = ex
    assert bad is not None, "unresolved vm_fd must be rejected"
    print(f"[7] prog_load relocation ok (prog={pid_prog['host_prog_id']})")

    # 9. spawn_cmd: POSIX_SPAWN_START_SUSPENDED cold-start hold
    spawn, _ = cli.call({
        "op": "spawn_cmd",
        "argv": [victim, "3"], "cwd": "/tmp",
        "env": {}, "hold_at_entry": True})
    spawn_id, sp_pid = spawn["spawn_id"], spawn["host_pid"]
    time.sleep(0.3)
    stat = subprocess.run(["ps", "-o", "stat=", "-p", str(sp_pid)],
                          capture_output=True, text=True).stdout.strip()
    assert stat.startswith("T"), f"held process not suspended: {stat}"
    pc, _ = cli.call({"op": "proc_check", "pid": sp_pid})
    assert pc["alive"] and victim in pc["exe"], pc
    print(f"[8a] spawned pid={sp_pid} suspended before dyld/main (stat={stat})")
    cli.call({"op": "cmd_ctl", "spawn_id": spawn_id, "action": "resume"})
    deadline = time.time() + 20
    exited = False
    while time.time() < deadline:
        st, out = cli.call({"op": "cmd_ctl", "spawn_id": spawn_id,
                            "action": "wait_poll"})
        if st["exited"]:
            exited = True
            assert st["exit_code"] == 0, st
            assert b"pid=" in out and b"done" in out, out
            break
        time.sleep(0.1)
    assert exited, "held victim did not exit after SIGCONT"
    print("[8b] resume via SIGCONT: victim ran to completion, output captured")

    # 10. spawn without hold, plus kill of a held long-running process
    spawn2, _ = cli.call({
        "op": "spawn_cmd", "argv": [victim, "2"], "cwd": "/tmp",
        "env": {}, "hold_at_entry": False})
    st, _ = cli.call({"op": "cmd_ctl", "spawn_id": spawn2["spawn_id"],
                      "action": "wait_poll"})
    deadline = time.time() + 20
    while not st["exited"] and time.time() < deadline:
        time.sleep(0.1)
        st, _ = cli.call({"op": "cmd_ctl", "spawn_id": spawn2["spawn_id"],
                          "action": "wait_poll"})
    assert st["exited"] and st["exit_code"] == 0, st
    spawn3, _ = cli.call({
        "op": "spawn_cmd", "argv": [victim, "100000", "loop"],
        "cwd": "/tmp", "env": {}, "hold_at_entry": True})
    cli.call({"op": "cmd_ctl", "spawn_id": spawn3["spawn_id"],
              "action": "kill", "sig": 15})
    deadline = time.time() + 20
    st = {"exited": False}
    while time.time() < deadline:
        st, _ = cli.call({"op": "cmd_ctl", "spawn_id": spawn3["spawn_id"],
                          "action": "wait_poll"})
        if st["exited"]:
            break
        time.sleep(0.1)
    assert st["exited"] and st["term_sig"] == 15, st
    print("[8c] unheld spawn and held-process kill/reap ok")

    # 11. subscribe_events: bounded 64-event ringbuf + perf-event streams
    sub = socket.create_connection(("127.0.0.1", port), timeout=30)
    send_frame(sub, {"op": "subscribe_events"})
    ack, _ = recv_frame(sub)
    assert ack["ok"], ack

    pe, _ = cli.call({
        "op": "perf_event_open", "kind": "software", "pid": -1,
        "cpu": 0, "sample_period": 0, "ref_ctr_off": 0})
    perf_fd = pe["host_perf_id"]

    subprocess.run([producer, shm_name, "ring", str(ring), str(EVENT_COUNT)],
                   check=True, timeout=30)
    subprocess.run([producer, shm_name, "perf", str(perf_fd),
                    str(EVENT_COUNT)], check=True, timeout=30)

    sub.settimeout(5)
    ring_events, perf_events = [], []
    deadline = time.time() + 15
    while time.time() < deadline and (
            len(ring_events) < EVENT_COUNT or len(perf_events) < EVENT_COUNT):
        try:
            meta, blob = recv_frame(sub)
        except socket.timeout:
            break
        if meta.get("op") == "ringbuf_event":
            assert meta["host_map_id"] == ring
            ring_events.append(struct.unpack("<Q", blob[:8])[0])
        elif meta.get("op") == "perf_event":
            assert meta["host_map_id"] == perf_fd
            perf_events.append(struct.unpack("<Q", blob[:8])[0])
    assert ring_events == list(range(EVENT_COUNT)), \
        f"ring events {len(ring_events)}"
    assert perf_events == list(range(EVENT_COUNT)), \
        f"perf events {len(perf_events)}"

    # Bounded stream: no further frames after both producers drain.
    sub.settimeout(1.5)
    extra = 0
    try:
        while True:
            recv_frame(sub)
            extra += 1
    except socket.timeout:
        pass
    assert extra == 0, f"stream not bounded: {extra} extra frames"
    print(f"[9] event stream ok: {EVENT_COUNT} ringbuf + {EVENT_COUNT} "
          f"perf records, in-order, bounded at 64/slice")
    sub.close()

    # 12. close_obj cleanup
    cli.call({"op": "close_obj", "host_id": perf_fd})
    cli.call({"op": "close_obj", "host_id": ring})
    cli.close()
    print("\nALL E2E CHECKS PASSED")
    return 0


def main():
    port = int(os.environ.get("BPFTIME_DARWIN_PORT", "49512"))
    victim = os.environ["BPFTIME_E2E_VICTIM"]
    producer = os.environ["BPFTIME_EVENT_PRODUCER"]
    host_bin = os.environ.get("BPFTIME_DARWIN_HOST_BIN")
    host_proc = None
    try:
        probe = socket.create_connection(("127.0.0.1", port), timeout=1)
        probe.close()
        print("[0] reusing already-running bpftime-darwin-host")
    except OSError:
        host_proc = subprocess.Popen(
            [host_bin, "-p", str(port)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for _ in range(100):
            try:
                probe = socket.create_connection(("127.0.0.1", port),
                                                 timeout=0.5)
                probe.close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            host_proc.wait(timeout=1)
            raise RuntimeError("bpftime-darwin-host failed to start:\n" +
                               host_proc.stdout.read())
        print(f"[0] started bpftime-darwin-host pid={host_proc.pid}")
    try:
        return run_tests(port, victim, producer)
    finally:
        if host_proc is not None:
            host_proc.terminate()
            try:
                host_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                host_proc.kill()


if __name__ == "__main__":
    sys.exit(main())
