/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026 wzycc
 *
 * True mid-function address probes via int3 breakpoints (x86_64).
 * See midfunction_int3.hpp for the rationale.
 */
#include "midfunction_int3.hpp"

#include <spdlog/spdlog.h>

#include <sys/mman.h>
#include <sys/signal.h>
#include <ucontext.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)

namespace bpftime {
namespace attach {

namespace {

/* SIGTRAP reports rip *after* the int3 byte, so the trap address is rip-1. */
inline uintptr_t trap_address(ucontext_t *uc)
{
	return (uintptr_t)uc->uc_mcontext.gregs[REG_RIP] - 1;
}

void convert_ucontext_to_pt_regs(ucontext_t *uc, pt_regs &regs)
{
	greg_t *g = uc->uc_mcontext.gregs;
	regs = {};
	regs.r15 = g[REG_R15];
	regs.r14 = g[REG_R14];
	regs.r13 = g[REG_R13];
	regs.r12 = g[REG_R12];
	regs.bp = g[REG_RBP];
	regs.bx = g[REG_RBX];
	regs.r11 = g[REG_R11];
	regs.r10 = g[REG_R10];
	regs.r9 = g[REG_R9];
	regs.r8 = g[REG_R8];
	regs.ax = g[REG_RAX];
	regs.cx = g[REG_RCX];
	regs.dx = g[REG_RDX];
	regs.si = g[REG_RSI];
	regs.di = g[REG_RDI];
	regs.orig_ax = (uint64_t)-1;
	regs.ip = (uint64_t)trap_address(uc);
	regs.cs = g[REG_CS];
	regs.flags = g[REG_EFL];
	regs.sp = g[REG_RSP];
	regs.ss = g[REG_SS];
}

} // namespace

struct midfunction_int3_registry::impl {
	std::mutex mtx;
	std::unordered_map<uintptr_t, std::vector<frida_internal_attach_entry *> >
		sites;
	bool handler_installed = false;
	struct sigaction prev_trap;

	void install_handler_once()
	{
		if (handler_installed)
			return;
		struct sigaction sa = {};
		sa.sa_sigaction = sigtrap_handler;
		sa.sa_flags = SA_SIGINFO;
		sigemptyset(&sa.sa_mask);
		sigaction(SIGTRAP, &sa, &prev_trap);
		handler_installed = true;
	}

	static void chain_to_prev(const struct sigaction *prev, int sig,
				  siginfo_t *info, void *uctx)
	{
		if (prev->sa_flags & SA_SIGINFO) {
			auto h = prev->sa_sigaction;
			if (h && (void (*)(int, siginfo_t *, void *))h != SIG_DFL &&
			    (void (*)(int, siginfo_t *, void *))h != SIG_IGN) {
				h(sig, info, uctx);
				return;
			}
		} else {
			auto h = prev->sa_handler;
			if (h && h != SIG_DFL && h != SIG_IGN) {
				h(sig);
				return;
			}
		}
		/* Nothing to chain to: restore the default disposition and
		 * return; the pending trap re-delivers with default action. */
		signal(sig, SIG_DFL);
	}

	static void sigtrap_handler(int sig, siginfo_t *info, void *uctx)
	{
		auto &self = *instance().p;
		ucontext_t *uc = (ucontext_t *)uctx;
		uintptr_t addr = trap_address(uc);

		std::vector<frida_internal_attach_entry *> entries;
		{
			std::lock_guard<std::mutex> lock(self.mtx);
			auto it = self.sites.find(addr);
			if (it == self.sites.end()) {
				chain_to_prev(&self.prev_trap, sig, info,
					      uctx);
				return;
			}
			entries = it->second;
		}

		pt_regs regs;
		convert_ucontext_to_pt_regs(uc, regs);
		for (auto *entry : entries)
			entry->iterate_uprobe_callbacks(regs);

		/* Resume after the int3 byte: the remainder of the nop sled
		 * (or, for a plain 0x90 site, the next instruction). */
		uc->uc_mcontext.gregs[REG_RIP] = (greg_t)(addr + 1);
	}

	static bool patch_int3(void *addr)
	{
		constexpr size_t page = 4096;
		uintptr_t a = (uintptr_t)addr;
		uintptr_t page_start = a & ~(page - 1);
		size_t len = page; /* the byte is within this one page */

		int prot_before = PROT_READ | PROT_EXEC;
		/* Query the current protection to restore exactly. */
		/* Best effort: try PROT_READ|PROT_WRITE|PROT_EXEC first. */
		if (mprotect((void *)page_start, len,
			     PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
			SPDLOG_ERROR(
				"midfunction int3: mprotect failed for {:x}: {}",
				page_start, strerror(errno));
			return false;
		}
		*(volatile uint8_t *)addr = 0xCC; /* INT3 */
		/* Leave the page executable; do not drop the write bit that
		 * gum or the loader may rely on - restore the common RX|W. */
		mprotect((void *)page_start, len,
			 PROT_READ | PROT_WRITE | PROT_EXEC);
		return true;
	}

	static void restore_byte(void *addr, uint8_t old)
	{
		constexpr size_t page = 4096;
		uintptr_t page_start = ((uintptr_t)addr) & ~(page - 1);
		mprotect((void *)page_start, page,
			 PROT_READ | PROT_WRITE | PROT_EXEC);
		*(volatile uint8_t *)addr = old;
	}
};

midfunction_int3_registry &midfunction_int3_registry::instance()
{
	static midfunction_int3_registry reg;
	if (reg.p == nullptr)
		reg.p = new impl();
	return reg;
}

bool midfunction_int3_registry::try_register(void *addr,
					     frida_internal_attach_entry *entry)
{
	/* Only take over single-byte-NOP-led sites: after resuming at addr+1
	 * the remaining bytes must still form valid, semantically identical
	 * instructions (the SDT nop sled guarantees this). */
	if (*(volatile uint8_t *)addr != 0x90)
		return false;

	auto *d = p;
	{
		std::lock_guard<std::mutex> lock(d->mtx);
		auto &vec = d->sites[(uintptr_t)addr];
		if (vec.empty()) {
			if (!d->patch_int3(addr)) {
				d->sites.erase((uintptr_t)addr);
				SPDLOG_WARN(
					"midfunction int3: unable to patch {:x}, falling back to frida function-entry attach",
					(uintptr_t)addr);
				return false;
			}
			d->install_handler_once();
			SPDLOG_INFO(
				"midfunction int3: instrumented note site {:x} (true mid-function trap)",
				(uintptr_t)addr);
		}
		vec.push_back(entry);
	}
	return true;
}

void midfunction_int3_registry::unregister(void *addr,
					   frida_internal_attach_entry *entry)
{
	auto *d = p;
	std::lock_guard<std::mutex> lock(d->mtx);
	auto it = d->sites.find((uintptr_t)addr);
	if (it == d->sites.end())
		return;
	auto &vec = it->second;
	for (size_t i = 0; i < vec.size(); i++) {
		if (vec[i] == entry) {
			vec.erase(vec.begin() + (long)i);
			break;
		}
	}
	if (vec.empty()) {
		d->restore_byte(addr, 0x90);
		d->sites.erase(it);
	}
}

bool midfunction_int3_registry::handles(void *addr) const
{
	if (p == nullptr)
		return false;
	std::lock_guard<std::mutex> lock(p->mtx);
	auto it = p->sites.find((uintptr_t)addr);
	return it != p->sites.end() && !it->second.empty();
}

} // namespace attach
} // namespace bpftime

#endif /* x86_64 */
