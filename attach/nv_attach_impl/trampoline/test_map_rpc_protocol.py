#!/usr/bin/env python3
"""Validate that multi-lane GPU map RPC publishes payloads under the lock."""

from pathlib import Path
import sys


def function_body(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[begin:finish]


def require_order(text: str, markers: list[str], context: str) -> None:
    position = 0
    for marker in markers:
        found = text.find(marker, position)
        if found < 0:
            raise AssertionError(f"{context}: missing or out-of-order {marker!r}")
        position = found + len(marker)


if len(sys.argv) != 3:
    raise SystemExit(f"usage: {sys.argv[0]} <trampoline.cu> <trampoline_ptx.h>")

source = Path(sys.argv[1]).read_text()
ptx = Path(sys.argv[2]).read_text()

rpc = function_body(
    source,
    'extern "C" __device__ HelperCallResponse make_map_helper_call(',
    "\n__device__ uint64_t getGlobalThreadId()",
)
require_order(
    rpc,
    [
        "spin_lock(&__bpftime_comm_lock)",
        "simple_memcpy(g_data->req.map_lookup.key",
        "simple_memcpy(g_data->req.map_update.key",
        "simple_memcpy(g_data->req.map_update.value",
        "simple_memcpy(g_data->req.map_delete.key",
        "complete_helper_call_locked",
        "spin_unlock(&__bpftime_comm_lock)",
        "__syncwarp(active_mask)",
    ],
    "map RPC critical section",
)

for helper, next_helper in (("0001", "0002"), ("0002", "0003"),
                            ("0003", "0006")):
    body = function_body(
        source,
        f"_bpf_helper_ext_{helper}(",
        f"_bpf_helper_ext_{next_helper}",
    )
    if "g_data->req" in body or "global_data->req" in body:
        raise AssertionError(
            f"map helper {helper} still publishes shared payload before locking"
        )
    if "make_map_helper_call" not in body:
        raise AssertionError(f"map helper {helper} bypasses serialized payload RPC")

if "// -- Begin function make_map_helper_call" not in ptx:
    raise AssertionError("generated PTX lacks the serialized map RPC function")
if "// -- Begin function _Z27complete_helper_call_locked" not in ptx:
    raise AssertionError("generated PTX lacks the locked helper handshake")

print("GPU map RPC source/PTX protocol validation passed")
