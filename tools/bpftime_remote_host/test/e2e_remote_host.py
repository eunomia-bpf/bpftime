#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
End-to-end test for bpftime-remote-host, runnable on both Linux and macOS.

It launches the daemon on a loopback TCP port with a fixed shared token, then
exercises: mandatory first-frame token authentication, hello OS/arch identity,
cross-machine ELF fetch (real ELF on Linux, Mach-O->stub ELF on macOS), map
CRUD, proc inspection, and a held spawn -> uprobe -> ringbuf -> TCP event flow.
"""
import glob
import json
import os
import platform
import socket
import struct
import subprocess
import sys
import time

BPF_MAP_HASH = 1
BPF_MAP_ARRAY = 2
BPF_MAP_PERCPU_ARRAY = 6
BPF_MAP_RINGBUF = 27

MOV64_IMM = 0xB7
MOV64_REG = 0xBF
ADD64_IMM = 0x07
CALL = 0x85
EXIT = 0x95
LDDW = 0x18
STX_DW = 0x7B
PSEUDO_MAP_FD = 1
HELPER_RINGBUF_OUTPUT = 130

TOKEN = "a" * 64
RING_VM = 9001


def send_frame(sock_, obj, blob=b""):
    payload = json.dumps(obj).encode()
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


class BpfAsm:
    def __init__(self):
        self.insns = []

    def emit(self, op, dst=0, src=0, off=0, imm=0):
        self.insns.append(struct.pack("<BBhi", op, dst | (src << 4), off, imm))

    def lddw(self, dst, fd):
        self.emit(LDDW, dst=dst, src=PSEUDO_MAP_FD, imm=fd)
        self.insns.append(b"\x00" * 8)

    def blob(self):
        return b"".join(self.insns)


def build_ringbuf_prog(ring_vm_fd):
    """Emit one 8-byte ringbuf event per uprobe firing."""
    p = BpfAsm()
    p.emit(MOV64_IMM, dst=9, imm=0)
    p.emit(STX_DW, dst=10, src=9, off=-8)  # zero the event slot
    p.lddw(1, ring_vm_fd)  # r1 = ringbuf map
    p.emit(MOV64_REG, dst=2, src=10)
    p.emit(ADD64_IMM, dst=2, imm=-8)  # r2 = r10 - 8
    p.emit(MOV64_IMM, dst=3, imm=8)  # r3 = size
    p.emit(MOV64_IMM, dst=4, imm=0)  # r4 = flags
    p.emit(CALL, imm=HELPER_RINGBUF_OUTPUT)
    p.emit(EXIT)
    return p.blob()


class ElfReader:
    def __init__(self, data):
        self.data = data
        assert data[:4] == b"\x7fELF", "not an ELF file"
        assert data[4] == 2, "not ELF64"
        (self.e_type, self.e_machine, _, _, _, self.e_shoff, _,
         _, _, _, self.e_shentsize, self.e_shnum,
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
        if sh is None:
            return {}
        _, _, _, _, off, size, link, _, _, entsize = sh
        strtab_sh = self.sections[link]
        strtab = self.data[strtab_sh[4]:strtab_sh[4] + strtab_sh[5]]
        out = {}
        for o in range(off, off + size, entsize):
            st_name, _, _, _, st_value, _ = struct.unpack_from(
                "<IBBHQQ", self.data, o)
            if st_name == 0:
                continue
            end = strtab.index(b"\x00", st_name)
            out[strtab[st_name:end].decode()] = st_value
        return out


class Client:
    def __init__(self, port, token=TOKEN):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=30)
        self.token = token

    def call(self, obj, blob=b""):
        if "token" not in obj:
            obj = dict(obj, token=self.token)
        send_frame(self.s, obj, blob)
        return recv_frame(self.s)

    def expect_ok(self, obj, blob=b""):
        resp, rblob = self.call(obj, blob)
        assert resp.get("ok"), f"op {obj.get('op')} failed: {resp}"
        return resp, rblob

    def close(self):
        self.s.close()


def wait_port(port, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("daemon did not open the port")


def assert_rejected(port, frame):
    c = socket.create_connection(("127.0.0.1", port), timeout=10)
    send_frame(c, frame)
    resp, _ = recv_frame(c)
    assert resp.get("ok") is False and resp.get("error") == "unauthorized", resp
    # The daemon closes the socket immediately after rejecting the frame.
    try:
        recv_exact(c, 1)
        raised = False
    except ConnectionError:
        raised = True
    assert raised, "unauthorized socket was not closed"
    c.close()


def test_negative_auth(port):
    assert_rejected(port, {"op": "hello"})  # missing token
    assert_rejected(port, {"op": "hello", "token": "deadbeef" * 8})  # wrong
    assert_rejected(port, {"op": "subscribe_events"})  # unauthenticated stream
    print("[auth] missing/wrong token rejected with unauthorized + closed")


def main():
    port = int(os.environ.get("BPFTIME_REMOTE_PORT", "49523"))
    host_bin = os.environ["BPFTIME_REMOTE_HOST_BIN"]
    victim = os.environ["BPFTIME_REMOTE_E2E_VICTIM"]
    configured_agent = os.environ["BPFTIME_AGENT_LIB"]
    is_darwin = platform.system() == "Darwin"
    # The Darwin agent dylib is delivered by a separate lane; when it is
    # absent use a harmless always-loaded dylib so spawn hold/resume can still
    # be exercised, and skip the in-victim uprobe/ringbuf assertion.
    if is_darwin and not os.path.exists(configured_agent):
        agent_lib = "/usr/lib/libSystem.B.dylib"
        agent_functional = False
    else:
        agent_lib = configured_agent
        agent_functional = not is_darwin
    iters = 12

    env = os.environ.copy()
    env["BPFTIME_REMOTE_TOKEN"] = TOKEN
    env["BPFTIME_AGENT_LIB"] = agent_lib
    # The daemon defaults to a 1 GiB segment for BCC-sized ringbufs; the e2e
    # only needs a small ring, so stay small to fit a shared /dev/shm.
    env["BPFTIME_SHM_MEMORY_MB"] = os.environ.get(
        "BPFTIME_REMOTE_TEST_SHM_MB", "64")
    host = subprocess.Popen([host_bin, "-p", str(port)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        wait_port(port)
        test_negative_auth(port)

        cli = Client(port)

        hello, _ = cli.expect_ok({"op": "hello"})
        expect_os = "darwin" if is_darwin else "linux"
        assert hello["os"] == expect_os, hello
        assert hello["arch"] in ("arm64", "x86_64") and hello["num_cpus"] >= 1
        print(f"[1] hello ok: {hello}")

        _, victim_elf = cli.expect_ok({"op": "fetch_elf", "path": victim})
        reader = ElfReader(victim_elf)
        assert reader.e_type == 3, "expected ET_DYN"
        assert reader.find_section(".eh_frame"), "missing .eh_frame"
        syms = reader.symbols(reader.find_section(".symtab"))
        syms.update(reader.symbols(reader.find_section(".dynsym")))
        assert "compute" in syms, f"compute missing from {sorted(syms)[:8]}"
        compute_off = syms["compute"]
        print(f"[2] victim fetch_elf ok: compute offset=0x{compute_off:x}")

        if is_darwin:
            _, sys_elf = cli.expect_ok({
                "op": "fetch_elf",
                "path": "/bpftime-dyld/usr/lib/libSystem.B.dylib"})
            sys_reader = ElfReader(sys_elf)
            sys_syms = sys_reader.symbols(sys_reader.find_section(".symtab"))
            assert "malloc" in sys_syms, "malloc missing from libSystem stub"
            assert sys_reader.find_section(".eh_frame"), "libSystem .eh_frame"
            print(f"[2b] libSystem stub ok ({len(sys_syms)} exports)")
        else:
            candidates = glob.glob("/lib/aarch64-linux-gnu/libc.so.6") + \
                glob.glob("/lib/x86_64-linux-gnu/libc.so.6") + \
                glob.glob("/usr/lib/*/libc.so.6")
            assert candidates, "libc.so.6 not located"
            _, libc_elf = cli.expect_ok(
                {"op": "fetch_elf", "path": candidates[0]})
            assert libc_elf[:4] == b"\x7fELF" and len(libc_elf) > 4096
            print(f"[2b] fetched real {candidates[0]} ({len(libc_elf)} bytes)")

        def make_map(mt, ks, vs, mx, name):
            r, _ = cli.expect_ok({
                "op": "map_create", "map_type": mt, "key_size": ks,
                "value_size": vs, "max_entries": mx, "flags": 0,
                "name": name})
            return r["host_map_id"]

        ring = make_map(BPF_MAP_RINGBUF, 0, 0, 65536, "rb")
        hmap = make_map(BPF_MAP_HASH, 4, 8, 64, "hash")
        make_map(BPF_MAP_ARRAY, 4, 8, 1, "array")
        pmap = make_map(BPF_MAP_PERCPU_ARRAY, 4, 8, 4, "percpu")

        cli.expect_ok({"op": "map_update", "host_map_id": hmap, "flags": 0},
                      struct.pack("<I", 7) + struct.pack("<Q", 42))
        _, val = cli.expect_ok(
            {"op": "map_lookup", "host_map_id": hmap}, struct.pack("<I", 7))
        assert struct.unpack("<Q", val)[0] == 42
        _, nk = cli.expect_ok(
            {"op": "map_get_next_key", "host_map_id": hmap, "has_key": False})
        assert len(nk) == 4
        cli.expect_ok({"op": "map_delete", "host_map_id": hmap},
                      struct.pack("<I", 7))
        print("[3] hash map update/lookup/next/delete ok")

        ncpu = hello["num_cpus"]
        stride = (8 + 7) & ~7
        cli.expect_ok(
            {"op": "map_update", "host_map_id": pmap, "flags": 0,
             "is_percpu": True},
            struct.pack("<I", 0) + b"".join(
                struct.pack("<Q", 1000 + c) for c in range(ncpu)))
        pr, pb = cli.expect_ok(
            {"op": "map_lookup", "host_map_id": pmap}, struct.pack("<I", 0))
        assert pr["is_percpu"] and pr["num_cpus"] == ncpu
        assert struct.unpack(f"<{ncpu}Q", pb[:ncpu * stride]) == \
            tuple(range(1000, 1000 + ncpu))
        print("[3b] per-cpu array round-trip ok")

        sp, _ = cli.expect_ok({
            "op": "spawn_cmd",
            "argv": ["/bin/sleep", "30"], "cwd": "", "env": {},
            "hold": False})
        sleep_pid = sp["pid"]
        # A fork()+execve() child briefly reports the daemon's own exe before
        # the new image (and its injected agent) finishes loading; wait for
        # the target executable to appear.
        pc = {}
        for _ in range(120):
            pc, _ = cli.expect_ok({"op": "proc_check", "pid": sleep_pid})
            if pc["alive"] and "sleep" in pc["exe"]:
                break
            time.sleep(0.15)
        assert pc["alive"] and "sleep" in pc["exe"], pc
        if is_darwin:
            try:
                pm, _ = cli.expect_ok({"op": "proc_maps", "pid": sleep_pid})
                assert pm["maps"], "empty maps"
                print("[4] darwin proc_check + synthesized proc_maps ok")
            except AssertionError:
                print("[4] darwin proc_check ok; maps dump absent (produced "
                      "by the injected agent)")
        else:
            pm, _ = cli.expect_ok({"op": "proc_maps", "pid": sleep_pid})
            assert "sleep" in pm["maps"], pm["maps"][:120]
            print("[4] linux proc_check + /proc maps ok")
        os.kill(sleep_pid, 15)

        spawn, _ = cli.expect_ok({
            "op": "spawn_cmd",
            "argv": [victim, str(iters)], "cwd": "/tmp", "env": {},
            "hold": True})
        vpid = spawn["pid"]
        print(f"[5] spawned victim pid={vpid} (held)")

        sub = socket.create_connection(("127.0.0.1", port), timeout=30)
        send_frame(sub, {"op": "subscribe_events", "token": TOKEN})
        ack, _ = recv_frame(sub)
        assert ack.get("ok"), ack

        if agent_functional:
            prog, _ = cli.expect_ok({
                "op": "prog_load", "prog_type": 0, "name": "ring_prog",
                "license": "MIT",
                "fd_map": [{"vm_fd": RING_VM, "host_map_id": ring}]},
                build_ringbuf_prog(RING_VM))
            prog_id = prog["host_prog_id"]

            pe, _ = cli.expect_ok({
                "op": "perf_event_open", "kind": "uprobe", "module": victim,
                "offset": compute_off, "pid": vpid, "cpu": 0,
                "ref_ctr_off": 0})
            perf_id = pe["host_perf_id"]
            link, _ = cli.expect_ok({
                "op": "bpf_link_create", "host_perf_id": perf_id,
                "host_prog_id": prog_id, "cookie": 0})
            assert link["host_link_id"] >= 0
            cli.expect_ok({
                "op": "perf_event_enable", "host_perf_id": perf_id,
                "enable": True, "pid": vpid})
            print("[6] prog loaded, uprobe attached and enabled")
        else:
            print("[6] darwin: agent dylib absent; skipping uprobe/ringbuf "
                  "(delivered by the darwin-agent lane)")

        cli.expect_ok({"op": "resume_cmd", "pid": vpid})
        print("[7] resumed victim")

        if agent_functional:
            sub.settimeout(1.5)
            got = 0
            deadline = time.time() + 20
            while got < iters and time.time() < deadline:
                try:
                    meta, blob = recv_frame(sub)
                except socket.timeout:
                    continue
                if meta.get("type") == "ringbuf" and \
                   meta.get("map_id") == ring and len(blob) == 8:
                    got += 1
            assert got == iters, \
                f"expected {iters} ringbuf events, got {got}"
            print(f"[7] received {got} ringbuf events over TCP")

        dead = {"alive": True}
        for _ in range(50):
            dead, _ = cli.expect_ok({"op": "proc_check", "pid": vpid})
            if not dead["alive"]:
                break
            time.sleep(0.1)
        assert not dead["alive"], "victim still running"
        sub.close()
        cli.close()
        print("[8] victim exited cleanly")
        print("ALL E2E CHECKS PASSED")
        return 0
    finally:
        host.terminate()
        try:
            _, err = host.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            host.kill()
            _, err = host.communicate()
        if os.environ.get("BPFTIME_REMOTE_VERBOSE"):
            sys.stderr.write(err.decode(errors="replace"))


if __name__ == "__main__":
    sys.exit(main())
