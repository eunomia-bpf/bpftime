#!/usr/bin/env python3
"""Validate the GPU ring-buffer memory protocol in source and generated PTX."""

from pathlib import Path
import sys


def require_order(text: str, markers: list[str], context: str) -> None:
    position = 0
    for marker in markers:
        next_position = text.find(marker, position)
        if next_position < 0:
            raise AssertionError(f"{context}: missing or out-of-order {marker!r}")
        position = next_position + len(marker)


if len(sys.argv) != 4:
    raise SystemExit(
        f"usage: {sys.argv[0]} <trampoline.cu> <trampoline_ptx.h> "
        "<bpf_attach_ctx.hpp>"
    )

source = Path(sys.argv[1]).read_text()
ptx = Path(sys.argv[2]).read_text()
host_header = Path(sys.argv[3]).read_text()
source_fence = 'asm volatile("membar.sys;" ::: "memory");'

for name, text in ("device", source), ("host", host_header):
    if "GPU_HELPER_MAX_BUF" not in text or "1 << 20" not in text:
        raise AssertionError(f"{name} GPU helper buffer is not 1 MiB")

ring_start = source.index("if (map_info.map_type == BPF_MAP_TYPE_GPU_RINGBUF_MAP)")
ring_end = source.index("\n\t} else {", ring_start)
ring_source = source[ring_start:ring_end]
require_order(
    ring_source,
    [
        "atomicCAS_system",
        source_fence,
        "const uint64_t head =",
        "volatile uint64_t",
        source_fence,
        "const uint64_t tail = header->tail",
    ],
    "source acquire path",
)

full_start = ring_source.index("if (tail - head")
full_end = ring_source.index("return 2;", full_start)
require_order(
    ring_source[full_start:full_end],
    ["errors->full_drops", source_fence, "header->dirty"],
    "source full-unlock path",
)

payload_start = ring_source.index("simple_memcpy", full_end)
require_order(
    ring_source[payload_start:],
    [
        "simple_memcpy",
        source_fence,
        "header->tail",
        source_fence,
        "header->dirty",
    ],
    "source publish/unlock path",
)

ptx_start = ptx.index("// -- Begin function _bpf_helper_ext_0025")
ptx_end = ptx.index("// -- End function", ptx_start)
ring_ptx = ptx[ptx_start:ptx_end]
if ring_ptx.count("membar.sys;") < 5:
    raise AssertionError("PTX ring helper lacks required system fences")

cas = ring_ptx.index("atom.sys.cas.b64")
acquire = ring_ptx.index("membar.sys;", cas)
head_load = ring_ptx.index("ld.volatile.u64", acquire)
head_acquire = ring_ptx.index("membar.sys;", head_load)
tail_load = ring_ptx.index("ld.u64", head_acquire)
if not cas < acquire < head_load < head_acquire < tail_load:
    raise AssertionError("PTX acquire/head-load ordering is invalid")
if "atom.sys.add.u64" in ring_ptx[acquire:head_acquire]:
    raise AssertionError("PTX reads CPU-owned head with a racing atomic RMW")

exchanges = []
position = 0
while True:
    position = ring_ptx.find("atom.sys.exch.b64", position)
    if position < 0:
        break
    exchanges.append(position)
    position += 1
if len(exchanges) < 3:
    raise AssertionError("PTX ring helper lacks full/tail/unlock atomics")

full_unlock, tail_publish, final_unlock = exchanges[-3:]
full_counter = ring_ptx.rfind("atom.sys.add.u64", 0, full_unlock)
full_release = ring_ptx.rfind("membar.sys;", 0, full_unlock)
if not full_counter < full_release < full_unlock:
    raise AssertionError("PTX full-path unlock lacks a release fence")

payload_publish = ring_ptx.rfind("membar.sys;", 0, tail_publish)
unlock_release = ring_ptx.rfind("membar.sys;", tail_publish, final_unlock)
if not payload_publish < tail_publish < unlock_release < final_unlock:
    raise AssertionError("PTX payload/tail/unlock ordering is invalid")

for required_offset in ("1048600", "2097176", "2097184"):
    if required_offset not in ptx:
        raise AssertionError(f"PTX missing 1 MiB ABI offset {required_offset}")
for stale_offset in ("16777240", "33554456"):
    if stale_offset in ptx:
        raise AssertionError(f"PTX still contains 16 MiB ABI offset {stale_offset}")

print("GPU ring-buffer source/PTX protocol validation passed")
