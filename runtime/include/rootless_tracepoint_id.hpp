/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef BPFTIME_ROOTLESS_TRACEPOINT_ID_HPP
#define BPFTIME_ROOTLESS_TRACEPOINT_ID_HPP

// ARCHITECTURE INVARIANT — DO NOT BREAK.
//
// This header is the single contract between the two rootless components
// that cooperate while tracefs is unavailable:
//
//   * runtime/syscall-server (producer) answers reads of tracepoint
//     ".../id" files and needs to synthesize stable ids;
//   * attach/syscall_trace_attach_impl (consumer) receives those ids and
//     must decode them before consulting any tracefs table or callback
//     array.
//
// Both components reach this header through the header-only
// bpftime_base_attach_impl target. They MUST NOT include each other's
// headers or link each other's libraries. In particular, the low-level
// syscall server must never depend on an attach implementation: the
// dependency direction is always producer/consumer -> this neutral
// contract, never one component -> the other.
//
// To keep that guarantee structural instead of conventional, everything
// here is constexpr arithmetic: no tables, no generated headers, no
// filesystem access, no logging. Each side owns the data it needs to map
// an event name to a syscall number; only the id encoding is shared.
//
// Kernel tracepoint ids are 16-bit values, so the reserved positive range
// starting at 0x40000000 can never collide with a genuine kernel id.

#include <cstdint>
#include <optional>

namespace bpftime
{
namespace rootless_tracepoint
{

// First reserved id. Raw syscall entry/exit occupy the first two slots;
// per-syscall events follow with one interleaved pair per syscall number.
inline constexpr int32_t SYNTHETIC_ID_BASE = 0x40000000;
inline constexpr int32_t RAW_SYS_ENTER_ID = SYNTHETIC_ID_BASE;
inline constexpr int32_t RAW_SYS_EXIT_ID = SYNTHETIC_ID_BASE + 1;

// Sched tracepoints are not syscalls and therefore cannot use the per-syscall
// interleaved encoding below. They get fixed high reserved slots so the
// syscall-server producer can answer their ".../id" reads and the attach
// consumer can map the id straight back to the sched event without a tracefs
// table. Kept far above the per-syscall range (which grows from the base) to
// avoid any collision.
inline constexpr int32_t SCHED_FORK_ID = 0x40010000;
inline constexpr int32_t SCHED_EXEC_ID = 0x40010001;
inline constexpr int32_t SCHED_EXIT_ID = 0x40010002;
struct decoded_syscall_event {
	// Kernel syscall number, or -1 for the raw syscall enter/exit
	// tracepoints that cover every syscall.
	int32_t sys_nr;
	// True for an enter event, false for an exit event.
	bool is_enter;
};

// Encode a concrete syscall number into the reserved range. Returns 0 when
// the number is outside the encodable range; callers treat 0 as "do not
// synthesize an id and fall back to the normal open path".
constexpr int32_t
encode_syscall_tracepoint_id(int32_t sys_nr, bool is_enter)
{
	if (sys_nr < 0 ||
	    sys_nr >= (INT32_MAX - SYNTHETIC_ID_BASE) / 2 - 1) {
		return 0;
	}
	return SYNTHETIC_ID_BASE + 2 * (sys_nr + 1) + (is_enter ? 0 : 1);
}

// Resolve a rootless sched tracepoint event path (relative to the
// ".../events/" root, e.g. "sched/sched_process_fork") to its fixed reserved
// id. Returns 0 for non-sched events so the caller falls back to the syscall
// encoding / normal open path. constexpr string-only, no table or FS access.
constexpr int32_t
sched_tracepoint_id_for_event(std::string_view event)
{
	if (event == "sched/sched_process_fork")
		return SCHED_FORK_ID;
	if (event == "sched/sched_process_exec")
		return SCHED_EXEC_ID;
	if (event == "sched/sched_process_exit")
		return SCHED_EXIT_ID;
	return 0;
}

// True when an id is one of the fixed sched reserved slots.
constexpr bool
is_rootless_sched_id(int32_t id)
{
	return id == SCHED_FORK_ID || id == SCHED_EXEC_ID ||
	       id == SCHED_EXIT_ID;
}
// True for any id in the reserved rootless range. A false result means the
// id is an ordinary kernel tracepoint id and the consumer must keep its
// existing tracefs-based resolution untouched.
constexpr bool
is_rootless_synthetic_id(int32_t id)
{
	return id >= SYNTHETIC_ID_BASE;
}

// Inverse of encode_syscall_tracepoint_id, also covering the two raw
// syscall slots. Returns std::nullopt for ids outside the reserved range;
// callers remain responsible for validating concrete syscall numbers
// against their own syscall table (the consumer side does this today).
constexpr std::optional<decoded_syscall_event>
decode_synthetic_tracepoint_id(int32_t id)
{
	if (!is_rootless_synthetic_id(id)) {
		return std::nullopt;
	}
	const int32_t offset = id - SYNTHETIC_ID_BASE;
	return decoded_syscall_event{
		.sys_nr = offset / 2 - 1,
		.is_enter = (offset % 2 == 0),
	};
}

} // namespace rootless_tracepoint
} // namespace bpftime

#endif // BPFTIME_ROOTLESS_TRACEPOINT_ID_HPP
