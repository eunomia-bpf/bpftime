/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026 wzycc
 *
 * True mid-function address probes via int3 breakpoints (x86_64).
 */
#ifndef _BPFTIME_MIDFUNCTION_INT3_HPP
#define _BPFTIME_MIDFUNCTION_INT3_HPP

#include "frida_internal_attach_entry.hpp"

namespace bpftime {
namespace attach {

#if defined(__x86_64__) || defined(_M_X64)

/*
 * frida-gum's interceptor only instruments function entries. An address in
 * the middle of a function is silently resolved to the *enclosing function's
 * entry*, so the attached program runs with entry-time register state. That
 * breaks USDT probes, whose notes live mid-function and whose argument specs
 * describe the machine state *at the note* (stack spill slots, register
 * allocation at that point).
 *
 * This registry restores kernel-uprobe semantics for such sites: when the
 * byte at the attach address is a single-byte NOP (0x90) - which is how the
 * SystemTap SDT nop sled begins on this toolchain - we overwrite that byte
 * with INT3 (0xCC) and dispatch callbacks from a SIGTRAP handler using the
 * pristine interrupted register state, then resume at addr+1 (the rest of
 * the nop sled), which is semantically identical to executing the sled.
 *
 * Callers whose target byte is not 0x90 keep the existing
 * (function-entry) behavior. Coverage is one-byte (0x90) sleds only:
 * a multi-byte NOP sled (0f 1f ...) starts with 0f, fails the check and
 * keeps the function-entry behavior - i.e. such sites keep the bug
 * described above. sys/sdt.h on this toolchain emits single-byte
 * sleds, so SDT notes are covered; there is no runtime warning on the
 * fallback path because ordinary function-entry uprobes take it too.
 *
 * fork(): a forked child inherits the patched byte, the SIGTRAP
 * disposition and this registry together (plain copy-on-write; the
 * registry is process-local, not in bpftime shared memory), so traps
 * in the child dispatch exactly as in the parent. A later detach in
 * either process restores only its own copy; exec*() discards all
 * three.
 */
class midfunction_int3_registry {
    public:
	static midfunction_int3_registry &instance();

	/* Try to take ownership of `addr` for `entry`. Returns true when the
	 * int3 path was installed (the caller must then skip the gum attach). */
	bool try_register(void *addr, class frida_internal_attach_entry *entry);

	void unregister(void *addr, class frida_internal_attach_entry *entry);

	bool handles(void *addr) const;

    private:
	struct impl;
	/* Takes ownership of nothing: `p` points at a function-local static
	 * in instance() (see there); pointer-to-incomplete-impl is fine. */
	explicit midfunction_int3_registry(impl *p) : p(p)
	{
	}
	impl *p = nullptr;
};

#else /* !x86_64 */

#include <spdlog/spdlog.h>

/* Stub so the call sites compile and link everywhere; never takes over.
 * instance() must be defined inline here: the .cpp defines it only
 * inside the x86_64 guard, while the call site builds on all arches -
 * a declaration-only stub would not link. */
class midfunction_int3_registry {
    public:
	static midfunction_int3_registry &instance()
	{
		static midfunction_int3_registry r;
		return r;
	}
	bool try_register(void *, void *)
	{
		SPDLOG_DEBUG(
			"midfunction int3 backend not implemented on this architecture; staying on function-entry attach");
		return false;
	}
	void unregister(void *, void *)
	{
	}
	bool handles(void *) const
	{
		return false;
	}

    private:
	midfunction_int3_registry() = default;
};

#endif

} // namespace attach
} // namespace bpftime

#endif
