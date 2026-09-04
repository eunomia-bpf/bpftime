/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "trap_uprobe_attach_impl.hpp"
#include "trap_attach_private_data.hpp"
#include "trap_arch.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <unwind.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <spdlog/spdlog.h>
#include <sys/mman.h>
#include <typeinfo>
#include <unistd.h>
#include <vector>

using namespace bpftime;
using namespace bpftime::attach;
using namespace bpftime::attach::trap;

namespace
{
constexpr size_t SLOT_SIZE = 32;
constexpr size_t URET_STACK_DEPTH = 128;
constexpr size_t MAX_BACKTRACE_FRAMES = 128;
// How far an out-of-line slot may be from the probed code.
constexpr intptr_t NEAR_DISTANCE = (intptr_t)1 << 30;

// A user facing attach entry. Owned by the engine so that an in-flight
// signal handler on another thread can never observe a destroyed entry.
struct attach_entry {
	int id;
	int type;
	uintptr_t function;
	trap_attach_impl *owner;
	attach_entry_callback callback;

	template <int callback_index> void run(const pt_regs &regs) const
	{
		struct func_ip_scope {
			uintptr_t previous;
			~func_ip_scope()
			{
				current_thread_attach_func_ip = previous;
			}
		} scope{ current_thread_attach_func_ip };
		if constexpr (callback_index == ATTACH_UPROBE_INDEX ||
			      callback_index == ATTACH_URETPROBE_INDEX)
			current_thread_attach_func_ip = function;
		else
			current_thread_attach_func_ip = 0;
		if (std::holds_alternative<callback_variant>(callback)) {
			std::get<callback_index>(
				std::get<callback_variant>(callback))(regs);
		} else {
			auto &args = std::get<ebpf_callback_args>(callback);
			uint64_t ret = 0;
			args.ebpf_cb((void *)&regs, sizeof(regs), &ret);
		}
	}
};

// State of one probed address. Immutable once published; modifications
// create a new probe_site that shares the slot and original bytes.
struct probe_site {
	uintptr_t addr = 0;
	arch::insn_info info;
	uint8_t orig[arch::MAX_INSN_LEN] = {};
	uint8_t trap[arch::MAX_TRAP_LEN] = {};
	size_t trap_len = 0;
	// Out-of-line slot, nullptr when the instruction is emulated
	uint8_t *slot = nullptr;
	uintptr_t slot_trap = 0;
	// Whether the trap instruction is currently written at `addr`
	bool armed = false;
	std::vector<attach_entry *> entries;
	bool has_override = false;
	bool has_uprobe = false;
	bool has_uretprobe = false;
};

struct site_table {
	// Sorted by addr
	std::vector<probe_site *> by_addr;
	// Sorted by slot_trap, only sites with a slot
	std::vector<probe_site *> by_slot_trap;

	probe_site *find_by_addr(uintptr_t addr) const
	{
		auto it = std::lower_bound(
			by_addr.begin(), by_addr.end(), addr,
			[](probe_site *s, uintptr_t a) { return s->addr < a; });
		return (it != by_addr.end() && (*it)->addr == addr) ? *it :
								       nullptr;
	}
	probe_site *find_by_slot_trap(uintptr_t pc) const
	{
		auto it = std::lower_bound(by_slot_trap.begin(),
					   by_slot_trap.end(), pc,
					   [](probe_site *s, uintptr_t a) {
						   return s->slot_trap < a;
					   });
		return (it != by_slot_trap.end() && (*it)->slot_trap == pc) ?
			       *it :
			       nullptr;
	}
};

struct uret_frame {
	uintptr_t function;
	uintptr_t orig_ret;
	uintptr_t sp;
};

struct uret_stack {
	uint32_t depth;
	uret_frame frames[URET_STACK_DEPTH];
};

struct override_state {
	bool is_overrided;
	uint64_t value;
	uint64_t ctx;
};

// Thread-local state touched from the signal handler.
//
// No tls_model attribute: this code is statically linked into a shared
// library (bpftime-agent) that is loaded via LD_PRELOAD or late dlopen
// into an arbitrary host process. initial-exec TLS requires static-TLS
// surplus which may already be exhausted; the default (global-dynamic)
// works in all loading scenarios.
//
// The uret_stack is embedded directly in TLS (~3 KB) so that the first
// uretprobe hit on any thread never calls mmap, malloc, or any other
// non-async-signal-safe function. The TLS block is reclaimed by the
// C runtime when the thread exits, so there is no VMA leak for
// short-lived threads.
thread_local int tl_in_handler = 0;
thread_local uret_stack tl_uret = {};
thread_local const pt_regs *tl_current_regs = nullptr;
thread_local uintptr_t tl_current_pc = 0;
thread_local uintptr_t tl_current_sp = 0;
thread_local int tl_phase = 0;
thread_local uint64_t tl_return_value = 0;
thread_local override_state tl_override = {};

// Number of SIGTRAP handlers currently executing, across all threads
std::atomic<int> active_handlers{ 0 };

uret_stack *get_uret_stack()
{
	return &tl_uret;
}

void set_override(uint64_t ctx, uint64_t value)
{
	tl_override.is_overrided = true;
	tl_override.value = value;
	tl_override.ctx = ctx;
}

struct code_region {
	uint8_t *base;
	size_t size;
	size_t used;
	bool writable;
};

std::string describe_target(uintptr_t addr)
{
	Dl_info info{};
	if (dladdr((void *)addr, &info) == 0 || !info.dli_fname)
		return "";
	std::string rendered = info.dli_fname;
	if (info.dli_sname) {
		rendered += "!";
		rendered += info.dli_sname;
		if (info.dli_saddr) {
			rendered += "+0x";
			char buf[32];
			snprintf(buf, sizeof(buf), "%lx",
				 (unsigned long)(addr - (uintptr_t)info.dli_saddr));
			rendered += buf;
		}
	}
	return rendered;
}

// The process wide engine: owns the SIGTRAP handler, every probe site and
// every attach entry ever created.
class trap_engine {
    public:
	static trap_engine &get()
	{
		static trap_engine instance;
		return instance;
	}

	int attach(trap_attach_impl *owner, int id, uintptr_t function,
		   attach_entry_callback &&cb, int type);
	int detach(int id, trap_attach_impl *owner);
	int detach_all_at(uintptr_t function, trap_attach_impl *owner,
			  std::vector<int> &removed_ids);
	void detach_owner(trap_attach_impl *owner);
	std::vector<uint64_t> *generate_stack();

    private:
	trap_engine();
	~trap_engine();

	static void on_sigtrap(int sig, siginfo_t *info, void *ctx);
	void handle_hit(probe_site *site, ucontext_t *uc);
	void handle_return(ucontext_t *uc);
	void chain_previous(int sig, siginfo_t *info, void *ctx);
	void resume(const probe_site *site, ucontext_t *uc);

	bool ensure_installed();
	int build_site(uintptr_t addr, probe_site &site, std::string &err);
	uint8_t *alloc_slot(uintptr_t near, bool &writable);
	bool finalize_slot(uint8_t *slot, bool writable, size_t len);
	bool write_code(uintptr_t addr, const uint8_t *data, size_t len);
	void publish(std::unique_ptr<site_table> table);
	void wait_for_quiescence();
	std::unique_ptr<site_table> copy_table_with(probe_site *replacement);

	std::mutex mu;
	std::atomic<site_table *> table{ nullptr };
	std::vector<std::unique_ptr<site_table>> tables;
	std::vector<std::unique_ptr<probe_site>> sites;
	// Every site that owns an out-of-line slot, sorted by slot_trap. A
	// thread may still be executing inside a slot long after the site
	// that created it was replaced, so slots stay resolvable forever.
	std::vector<probe_site *> slot_sites;
	std::vector<std::unique_ptr<attach_entry>> entries;
	std::vector<code_region> regions;
	uintptr_t uret_trampoline = 0;
	struct sigaction previous_action {};
	bool installed = false;
};

trap_engine::trap_engine() = default;

trap_engine::~trap_engine()
{
	if (installed) {
		struct sigaction current {};
		if (sigaction(SIGTRAP, nullptr, &current) == 0 &&
		    current.sa_sigaction == &trap_engine::on_sigtrap) {
			sigaction(SIGTRAP, &previous_action, nullptr);
		}
	}
}

bool trap_engine::ensure_installed()
{
	if (uret_trampoline == 0) {
		// Trampoline hit by the hijacked return address of uretprobes
		uint8_t trap[arch::MAX_TRAP_LEN];
		size_t trap_len = arch::trap_bytes(4, trap);
		bool writable;
		uint8_t *tramp = alloc_slot(0, writable);
		if (!tramp) {
			SPDLOG_ERROR("Unable to allocate uretprobe trampoline");
			return false;
		}
		std::memcpy(tramp, trap, trap_len);
		if (!finalize_slot(tramp, writable, trap_len))
			return false;
		uret_trampoline = (uintptr_t)tramp;
	}
	// The host may have (re)installed its own SIGTRAP handler after we
	// installed ours. Put ours back in front and forward to the host's.
	struct sigaction current {};
	if (sigaction(SIGTRAP, nullptr, &current) == 0 &&
	    (current.sa_flags & SA_SIGINFO) &&
	    current.sa_sigaction == &trap_engine::on_sigtrap)
		return true;
	if (installed) {
		SPDLOG_INFO(
			"SIGTRAP handler was replaced by the host process, re-installing the trap uprobe handler in front of it");
	}

	struct sigaction sa {};
	sa.sa_sigaction = &trap_engine::on_sigtrap;
	sigemptyset(&sa.sa_mask);
	// SA_NODEFER: a probed function invoked from inside a callback must
	// still be able to trap, otherwise the kernel kills the process.
	sa.sa_flags = SA_SIGINFO | SA_NODEFER | SA_RESTART;
	if (sigaction(SIGTRAP, &sa, &previous_action) != 0) {
		SPDLOG_ERROR("Unable to install SIGTRAP handler: {}",
			     strerror(errno));
		return false;
	}
	installed = true;
	SPDLOG_DEBUG("Installed trap uprobe SIGTRAP handler ({})",
		     arch::name());
	return true;
}

uint8_t *trap_engine::alloc_slot(uintptr_t near, bool &writable)
{
	const size_t page = (size_t)sysconf(_SC_PAGESIZE);
	for (auto &r : regions) {
		if (r.used + SLOT_SIZE > r.size)
			continue;
		if (near != 0) {
			intptr_t dist = (intptr_t)r.base - (intptr_t)near;
			if (dist > NEAR_DISTANCE || dist < -NEAR_DISTANCE)
				continue;
		}
		uint8_t *slot = r.base + r.used;
		r.used += SLOT_SIZE;
		writable = r.writable;
		return slot;
	}
	auto try_map = [&](void *hint, int flags, int prot) -> void * {
		void *p = mmap(hint, page, prot,
			       MAP_PRIVATE | MAP_ANONYMOUS | flags, -1, 0);
		if (p == MAP_FAILED)
			return nullptr;
		if (hint && p != hint) {
			intptr_t dist = (intptr_t)p - (intptr_t)near;
			if (dist > NEAR_DISTANCE || dist < -NEAR_DISTANCE) {
				munmap(p, page);
				return nullptr;
			}
		}
		return p;
	};
	void *mem = nullptr;
	bool rwx = true;
	int rwx_prot = PROT_READ | PROT_WRITE | PROT_EXEC;
	if (near != 0) {
		uintptr_t base = near & ~(uintptr_t)(page - 1);
		for (intptr_t k = 1; k <= 64 && !mem; k++) {
			for (int sign : { 1, -1 }) {
				intptr_t off = sign * k * ((intptr_t)16 << 20);
				uintptr_t hint = (uintptr_t)((intptr_t)base + off);
				if (hint < page)
					continue;
#ifdef MAP_FIXED_NOREPLACE
				mem = try_map((void *)hint, MAP_FIXED_NOREPLACE,
					      rwx_prot);
#else
				mem = try_map((void *)hint, 0, rwx_prot);
#endif
				if (mem)
					break;
			}
		}
	}
	if (!mem)
		mem = try_map(nullptr, 0, rwx_prot);
	if (!mem) {
		// W^X policy: fall back to a read-write page that is made
		// executable once written. One slot per page, since other
		// threads may already be executing earlier slots.
		mem = try_map(nullptr, 0, PROT_READ | PROT_WRITE);
		rwx = false;
	}
	if (!mem) {
		SPDLOG_ERROR("Unable to allocate executable memory: {}",
			     strerror(errno));
		return nullptr;
	}
	regions.push_back(code_region{ (uint8_t *)mem, rwx ? page : SLOT_SIZE,
				       SLOT_SIZE, rwx });
	writable = rwx;
	return (uint8_t *)mem;
}

bool trap_engine::finalize_slot(uint8_t *slot, bool writable, size_t len)
{
	arch::flush_icache(slot, len);
	if (!writable) {
		const size_t page = (size_t)sysconf(_SC_PAGESIZE);
		void *base = (void *)((uintptr_t)slot & ~(uintptr_t)(page - 1));
		if (mprotect(base, page, PROT_READ | PROT_EXEC) != 0) {
			SPDLOG_ERROR("Unable to make slot executable: {}",
				     strerror(errno));
			return false;
		}
	}
	return true;
}

// Patch `len` bytes of code at `addr`. The store is done with a single
// aligned write of the natural size so that other threads observe either
// the old or the new instruction.
bool trap_engine::write_code(uintptr_t addr, const uint8_t *data, size_t len)
{
	const size_t page = (size_t)sysconf(_SC_PAGESIZE);
	uintptr_t start = addr & ~(uintptr_t)(page - 1);
	uintptr_t end = (addr + len + page - 1) & ~(uintptr_t)(page - 1);
	int restore_prot = PROT_READ | PROT_EXEC;
	if (mprotect((void *)start, end - start,
		     PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
		if (mprotect((void *)start, end - start,
			     PROT_READ | PROT_WRITE) != 0) {
			SPDLOG_ERROR("Unable to make {:x} writable: {}", addr,
				     strerror(errno));
			return false;
		}
	}
	if (len == 1) {
		__atomic_store_n((uint8_t *)addr, data[0], __ATOMIC_SEQ_CST);
	} else if (len == 2 && (addr & 1) == 0) {
		uint16_t v;
		std::memcpy(&v, data, 2);
		__atomic_store_n((uint16_t *)addr, v, __ATOMIC_SEQ_CST);
	} else if (len == 4 && (addr & 3) == 0) {
		uint32_t v;
		std::memcpy(&v, data, 4);
		__atomic_store_n((uint32_t *)addr, v, __ATOMIC_SEQ_CST);
	} else if (len == 4 && (addr & 1) == 0) {
		// 4-byte instruction at a 2-byte boundary (RISC-V with C
		// extension).  Two 16-bit stores are needed.  The intermediate
		// state must be a trapping instruction no matter which half
		// another hart fetches first, so we go through c.ebreak
		// (0x9002) which is a complete compressed trap:
		//   Phase 1: write c.ebreak into the low half — the site now
		//            always traps regardless of the high half.
		//   Phase 2: write the intended high half (c.ebreak still
		//            guards the low half).
		//   Phase 3: write the intended low half, completing the
		//            transition atomically from the fetch perspective.
		constexpr uint16_t C_EBREAK = 0x9002;
		uint16_t lo, hi;
		std::memcpy(&lo, data, 2);
		std::memcpy(&hi, data + 2, 2);
		__atomic_store_n((uint16_t *)addr, C_EBREAK,
				 __ATOMIC_RELEASE);
		arch::flush_icache((void *)addr, 4);
		__atomic_store_n((uint16_t *)(addr + 2), hi,
				 __ATOMIC_RELEASE);
		arch::flush_icache((void *)addr, 4);
		__atomic_store_n((uint16_t *)addr, lo, __ATOMIC_RELEASE);
	} else {
		std::memcpy((void *)addr, data, len);
	}
	arch::flush_icache((void *)addr, len);
	if (mprotect((void *)start, end - start, restore_prot) != 0) {
		SPDLOG_WARN("Unable to restore protection of {:x}: {}", addr,
			    strerror(errno));
	}
	return true;
}

int trap_engine::build_site(uintptr_t addr, probe_site &site, std::string &err)
{
	site.addr = addr;
	auto info = arch::decode((const uint8_t *)addr, err);
	if (!info)
		return -ENOTSUP;
	site.info = *info;
	std::memcpy(site.orig, (const void *)addr, site.info.len);
	site.trap_len = arch::trap_bytes(site.info.len, site.trap);
	if (site.info.kind == arch::insn_kind::execute_out_of_line) {
		bool writable;
		uint8_t *slot = alloc_slot(addr, writable);
		if (!slot) {
			err = "unable to allocate an out-of-line slot";
			return -ENOMEM;
		}
		size_t trap_offset;
		// The slot is writable at this point: either the region is
		// mapped RWX, or it is a fresh RW page that finalize_slot
		// turns into RX below.
		if (!arch::prepare_out_of_line(site.orig, site.info, addr, slot,
					       SLOT_SIZE, &trap_offset, err))
			return -ENOTSUP;
		if (!finalize_slot(slot, writable, SLOT_SIZE)) {
			err = "unable to make the out-of-line slot executable";
			return -EIO;
		}
		site.slot = slot;
		site.slot_trap = (uintptr_t)slot + trap_offset;
	}
	return 0;
}

std::unique_ptr<site_table>
trap_engine::copy_table_with(probe_site *replacement)
{
	auto next = std::make_unique<site_table>();
	site_table *cur = table.load(std::memory_order_acquire);
	if (cur) {
		for (auto *s : cur->by_addr) {
			if (s->addr != replacement->addr)
				next->by_addr.push_back(s);
		}
	}
	next->by_addr.push_back(replacement);
	std::sort(next->by_addr.begin(), next->by_addr.end(),
		  [](probe_site *a, probe_site *b) { return a->addr < b->addr; });
	next->by_slot_trap = slot_sites;
	return next;
}

void trap_engine::publish(std::unique_ptr<site_table> next)
{
	table.store(next.get(), std::memory_order_release);
	tables.push_back(std::move(next));
}

// After a table has been published, handlers that loaded the previous table
// may still be running callbacks of entries that were just removed. Wait
// (bounded) until every such handler has finished so that a caller of
// detach can assume its callbacks are no longer invoked.
void trap_engine::wait_for_quiescence()
{
	const int self = tl_in_handler ? 1 : 0;
	for (int spins = 0; spins < 100000; spins++) {
		if (active_handlers.load(std::memory_order_acquire) <= self)
			return;
		sched_yield();
	}
	SPDLOG_WARN(
		"trap uprobe: handlers still running after detach, callbacks may fire once more");
}

int trap_engine::attach(trap_attach_impl *owner, int id, uintptr_t function,
			attach_entry_callback &&cb, int type)
{
	std::lock_guard<std::mutex> guard(mu);
	if (!ensure_installed())
		return -EIO;
	site_table *cur = table.load(std::memory_order_acquire);
	probe_site *existing = cur ? cur->find_by_addr(function) : nullptr;
	bool reuse = existing && existing->armed;
	if (reuse) {
		if (existing->has_override) {
			SPDLOG_ERROR(
				"Function {:x} was already attached with replace or filter, cannot attach anything else",
				function);
			return -EEXIST;
		}
		if (type == ATTACH_UPROBE_OVERRIDE) {
			SPDLOG_ERROR(
				"Function {:x} already has uprobe/uretprobe attaches, cannot attach a filter or replace",
				function);
			return -EEXIST;
		}
	}
	auto site = std::make_unique<probe_site>();
	bool new_slot = false;
	if (reuse) {
		*site = *existing;
	} else if (existing &&
		   std::memcmp((const void *)function, existing->orig,
			       existing->info.len) == 0) {
		// Re-arming a previously detached address whose code did not
		// change: keep its decoded state and out-of-line slot
		*site = *existing;
		site->entries.clear();
		site->has_override = site->has_uprobe = site->has_uretprobe =
			false;
	} else {
		std::string err;
		if (int res = build_site(function, *site, err); res < 0) {
			SPDLOG_ERROR(
				"Unable to place a trap uprobe at 0x{:x} ({}): {}",
				function, describe_target(function), err);
			return res;
		}
		new_slot = site->slot != nullptr;
	}
	auto entry = std::make_unique<attach_entry>();
	entry->id = id;
	entry->type = type;
	entry->function = function;
	entry->owner = owner;
	entry->callback = std::move(cb);
	site->entries.push_back(entry.get());
	site->has_override = site->has_override ||
			     type == ATTACH_UPROBE_OVERRIDE;
	site->has_uprobe = site->has_uprobe || type == ATTACH_UPROBE;
	site->has_uretprobe = site->has_uretprobe || type == ATTACH_URETPROBE;
	site->armed = true;

	probe_site *site_ptr = site.get();
	if (new_slot) {
		auto pos = std::lower_bound(slot_sites.begin(), slot_sites.end(),
					    site_ptr->slot_trap,
					    [](probe_site *a, uintptr_t v) {
						    return a->slot_trap < v;
					    });
		slot_sites.insert(pos, site_ptr);
	}
	publish(copy_table_with(site_ptr));
	sites.push_back(std::move(site));
	entries.push_back(std::move(entry));
	if (!reuse) {
		// Arm after publishing so that a hit always finds its site
		if (!write_code(function, site_ptr->trap, site_ptr->trap_len)) {
			site_ptr->armed = false;
			return -EIO;
		}
		SPDLOG_DEBUG("Armed trap uprobe at {:x} ({} bytes, {})",
			     function, site_ptr->trap_len,
			     site_ptr->slot ? "out-of-line" : "emulated");
	}
	return id;
}

// Remove `remove` from the site of `function`. Called with `mu` held.
static void remove_entries_from_site(probe_site &site,
				     const std::vector<attach_entry *> &remove)
{
	site.entries.erase(std::remove_if(site.entries.begin(),
					  site.entries.end(),
					  [&](attach_entry *e) {
						  return std::find(remove.begin(),
								   remove.end(),
								   e) !=
							 remove.end();
					  }),
			   site.entries.end());
	site.has_override = site.has_uprobe = site.has_uretprobe = false;
	for (auto *e : site.entries) {
		site.has_override |= e->type == ATTACH_UPROBE_OVERRIDE;
		site.has_uprobe |= e->type == ATTACH_UPROBE;
		site.has_uretprobe |= e->type == ATTACH_URETPROBE;
	}
}

int trap_engine::detach(int id, trap_attach_impl *owner)
{
	std::lock_guard<std::mutex> guard(mu);
	site_table *cur = table.load(std::memory_order_acquire);
	if (!cur)
		return -ENOENT;
	for (auto *s : cur->by_addr) {
		for (auto *e : s->entries) {
			if (e->id != id || e->owner != owner)
				continue;
			auto site = std::make_unique<probe_site>(*s);
			remove_entries_from_site(*site, { e });
			if (site->entries.empty()) {
				write_code(site->addr, site->orig,
					   site->trap_len);
				site->armed = false;
				SPDLOG_DEBUG("Disarmed trap uprobe at {:x}",
					     site->addr);
			}
			publish(copy_table_with(site.get()));
			sites.push_back(std::move(site));
			wait_for_quiescence();
			return 0;
		}
	}
	return -ENOENT;
}

int trap_engine::detach_all_at(uintptr_t function, trap_attach_impl *owner,
			       std::vector<int> &removed_ids)
{
	std::lock_guard<std::mutex> guard(mu);
	site_table *cur = table.load(std::memory_order_acquire);
	probe_site *s = cur ? cur->find_by_addr(function) : nullptr;
	if (!s || !s->armed)
		return -ENOENT;
	std::vector<attach_entry *> remove;
	for (auto *e : s->entries) {
		if (e->owner == owner) {
			remove.push_back(e);
			removed_ids.push_back(e->id);
		}
	}
	if (remove.empty())
		return -ENOENT;
	auto site = std::make_unique<probe_site>(*s);
	remove_entries_from_site(*site, remove);
	if (site->entries.empty()) {
		write_code(site->addr, site->orig, site->trap_len);
		site->armed = false;
	}
	publish(copy_table_with(site.get()));
	sites.push_back(std::move(site));
	wait_for_quiescence();
	return 0;
}

void trap_engine::detach_owner(trap_attach_impl *owner)
{
	std::vector<uintptr_t> functions;
	{
		std::lock_guard<std::mutex> guard(mu);
		site_table *cur = table.load(std::memory_order_acquire);
		if (!cur)
			return;
		for (auto *s : cur->by_addr) {
			for (auto *e : s->entries) {
				if (e->owner == owner) {
					functions.push_back(s->addr);
					break;
				}
			}
		}
	}
	for (auto f : functions) {
		std::vector<int> ids;
		detach_all_at(f, owner, ids);
	}
}

void trap_engine::resume(const probe_site *site, ucontext_t *uc)
{
	if (site->slot) {
		arch::set_pc(uc, (uintptr_t)site->slot);
	} else {
		arch::emulate(uc, site->orig, site->info, site->addr);
	}
}

void trap_engine::handle_hit(probe_site *site, ucontext_t *uc)
{
	if (tl_in_handler == 0 && site->armed && !site->entries.empty()) {
		tl_in_handler = 1;
		pt_regs regs;
		arch::fill_pt_regs(uc, site->addr, regs);
		tl_current_regs = &regs;
		tl_current_pc = site->addr;
		tl_current_sp = arch::get_sp(uc);
		tl_phase = 1;
		bool overrided = false;
		try {
			if (site->has_override) {
				tl_override = override_state{};
				curr_thread_override_return_callback =
					override_return_set_callback(
						set_override);
				for (auto *e : site->entries) {
					if (e->type == ATTACH_UPROBE_OVERRIDE) {
						e->run<ATTACH_UPROBE_OVERRIDE_INDEX>(
							regs);
						break;
					}
				}
				curr_thread_override_return_callback.reset();
				overrided = tl_override.is_overrided;
			} else {
				for (auto *e : site->entries) {
					if (e->type == ATTACH_UPROBE)
						e->run<ATTACH_UPROBE_INDEX>(regs);
				}
			}
		} catch (...) {
		}
		tl_phase = 0;
		tl_current_regs = nullptr;
		if (overrided) {
			tl_in_handler = 0;
			arch::do_return(uc, tl_override.value);
			return;
		}
		if (site->has_uretprobe) {
			uret_stack *stack = get_uret_stack();
			if (stack->depth < URET_STACK_DEPTH) {
				auto &frame = stack->frames[stack->depth++];
				frame.function = site->addr;
				frame.orig_ret = arch::get_return_address(uc);
				frame.sp = arch::get_sp(uc);
				arch::set_return_address(uc, uret_trampoline);
			}
		}
		tl_in_handler = 0;
	}
	resume(site, uc);
}

void trap_engine::handle_return(ucontext_t *uc)
{
	uret_stack *stack = &tl_uret;
	uintptr_t sp = arch::get_sp(uc);
	// Drop frames abandoned by longjmp/exceptions: their stack pointer is
	// below the one we are returning with.
	while (stack->depth > 0 &&
	       stack->frames[stack->depth - 1].sp + sizeof(uintptr_t) < sp)
		stack->depth--;
	if (stack->depth == 0) {
		signal(SIGTRAP, SIG_DFL);
		return;
	}
	uret_frame frame = stack->frames[--stack->depth];
	arch::set_pc(uc, frame.orig_ret);
	if (tl_in_handler != 0)
		return;
	site_table *cur = table.load(std::memory_order_acquire);
	probe_site *site = cur ? cur->find_by_addr(frame.function) : nullptr;
	if (!site || !site->armed || !site->has_uretprobe)
		return;
	tl_in_handler = 1;
	pt_regs regs;
	arch::fill_pt_regs(uc, frame.orig_ret, regs);
	tl_current_regs = &regs;
	tl_current_pc = frame.orig_ret;
	tl_current_sp = arch::get_sp(uc);
	tl_phase = 2;
	tl_return_value = arch::get_return_value(uc);
	try {
		for (auto *e : site->entries) {
			if (e->type == ATTACH_URETPROBE)
				e->run<ATTACH_URETPROBE_INDEX>(regs);
		}
	} catch (...) {
	}
	tl_phase = 0;
	tl_current_regs = nullptr;
	tl_in_handler = 0;
}

void trap_engine::chain_previous(int sig, siginfo_t *info, void *ctx)
{
	if (previous_action.sa_flags & SA_SIGINFO) {
		if (previous_action.sa_sigaction)
			previous_action.sa_sigaction(sig, info, ctx);
		return;
	}
	if (previous_action.sa_handler == SIG_IGN)
		return;
	if (previous_action.sa_handler == SIG_DFL) {
		// Not our trap: restore the default disposition and re-raise
		// so the process terminates the way it would have without us.
		sigaction(SIGTRAP, &previous_action, nullptr);
		raise(SIGTRAP);
		return;
	}
	previous_action.sa_handler(sig);
}

void trap_engine::on_sigtrap(int sig, siginfo_t *info, void *ctx)
{
	int saved_errno = errno;
	auto *uc = (ucontext_t *)ctx;
	trap_engine &engine = trap_engine::get();
	active_handlers.fetch_add(1, std::memory_order_acq_rel);
	uintptr_t pc = arch::trap_pc(uc);
	if (pc == engine.uret_trampoline && pc != 0) {
		engine.handle_return(uc);
	} else if (site_table *cur =
			   engine.table.load(std::memory_order_acquire);
		   cur != nullptr) {
		if (probe_site *s = cur->find_by_slot_trap(pc); s) {
			// Finished executing the relocated instruction
			arch::set_pc(uc, s->addr + s->info.len);
		} else if (probe_site *hit = cur->find_by_addr(pc); hit) {
			engine.handle_hit(hit, uc);
		} else {
			engine.chain_previous(sig, info, ctx);
		}
	} else {
		engine.chain_previous(sig, info, ctx);
	}
	active_handlers.fetch_sub(1, std::memory_order_acq_rel);
	errno = saved_errno;
}

struct unwind_state {
	uintptr_t interrupted_sp;
	uintptr_t interrupted_pc;
	std::vector<uint64_t> *frames;
	bool past_handler;
};

// _Unwind_Backtrace callback. Inside the callback _Unwind_GetCFA() yields
// the stack pointer of the frame being visited (the CFA of its callee), so
// frames of our handler and the signal trampoline report values below the
// interrupted stack pointer, the interrupted function reports the address
// of the signal context (also below it), and the interrupted function's
// callers report values at or above it. Only the callers are collected;
// the interrupted pc itself is reported as frame 0 by the caller of this.
_Unwind_Reason_Code collect_frame(struct _Unwind_Context *ctx, void *arg)
{
	auto *st = (unwind_state *)arg;
	uintptr_t cfa = (uintptr_t)_Unwind_GetCFA(ctx);
	if (cfa < st->interrupted_sp)
		return _URC_NO_REASON;
	uintptr_t ip = (uintptr_t)_Unwind_GetIP(ctx);
	if (!st->past_handler) {
		st->past_handler = true;
		if (ip == st->interrupted_pc || ip == st->interrupted_pc + 1)
			return _URC_NO_REASON;
	}
	st->frames->push_back((uint64_t)ip);
	return st->frames->size() >= MAX_BACKTRACE_FRAMES ? _URC_END_OF_STACK :
							     _URC_NO_REASON;
}

// Stack of the interrupted thread as seen by the probe: frame 0 is the
// interrupted pc (the probed function at entry, the return address in its
// caller at exit), followed by the return addresses of the callers. This
// mirrors what the kernel reports for bpf_get_stack on a uprobe.
std::vector<uint64_t> *trap_engine::generate_stack()
{
	if (tl_current_regs == nullptr) {
		SPDLOG_ERROR("There is no trap uprobe running");
		return nullptr;
	}
	auto result = new std::vector<uint64_t>;
	result->push_back(tl_current_pc);
	unwind_state st{ tl_current_sp, tl_current_pc, result, false };
	_Unwind_Backtrace(collect_frame, &st);
	return result;
}
} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void trap_attach_impl::prepare_thread()
{
	// No-op: uret_stack is now embedded in TLS and always available.
	// Kept for API compatibility.
}

int bpftime::attach::trap::from_cb_idx_to_attach_type(int idx)
{
	switch (idx) {
	case ATTACH_UPROBE_INDEX:
		return ATTACH_UPROBE;
	case ATTACH_URETPROBE_INDEX:
		return ATTACH_URETPROBE;
	case ATTACH_UPROBE_OVERRIDE_INDEX:
		return ATTACH_UPROBE_OVERRIDE;
	default:
		SPDLOG_ERROR("Unreachable branch reached!");
		return -1;
	}
}

std::optional<std::string>
bpftime::attach::trap::check_probe_target(const void *func_addr)
{
	if (func_addr == nullptr)
		return "address is null";
	std::string err;
	if (!arch::decode((const uint8_t *)func_addr, err))
		return err;
	return std::nullopt;
}

trap_attach_impl::trap_attach_impl()
{
	SPDLOG_DEBUG("Initializing trap uprobe attach impl ({})", arch::name());
	// Force the engine to exist so its constructor runs outside of any
	// signal handler
	(void)trap_engine::get();
}

trap_attach_impl::~trap_attach_impl()
{
	trap_engine::get().detach_owner(this);
	attaches.clear();
}

int trap_attach_impl::attach_at(void *func_addr, attach_entry_callback &&cb)
{
	if (func_addr == nullptr) {
		SPDLOG_ERROR("Unable to attach uprobes to address 0");
		return -EINVAL;
	}
	int type;
	if (std::holds_alternative<callback_variant>(cb))
		type = from_cb_idx_to_attach_type(
			(int)std::get<callback_variant>(cb).index());
	else
		type = std::get<ebpf_callback_args>(cb).attach_type;
	int id = allocate_id();
	int res = trap_engine::get().attach(this, id, (uintptr_t)func_addr,
					    std::move(cb), type);
	if (res < 0)
		return res;
	attaches[id] = owned_entry{ (uintptr_t)func_addr, type };
	return id;
}

int trap_attach_impl::create_uprobe_at(void *func_addr, uprobe_callback &&cb)
{
	return attach_at(func_addr,
			 callback_variant(
				 std::in_place_index_t<ATTACH_UPROBE_INDEX>(),
				 std::move(cb)));
}

int trap_attach_impl::create_uretprobe_at(void *func_addr,
					  uretprobe_callback &&cb)
{
	return attach_at(
		func_addr,
		callback_variant(std::in_place_index_t<ATTACH_URETPROBE_INDEX>(),
				 std::move(cb)));
}

int trap_attach_impl::create_uprobe_override_at(void *func_addr,
						uprobe_override_callback &&cb)
{
	return attach_at(
		func_addr,
		callback_variant(
			std::in_place_index_t<ATTACH_UPROBE_OVERRIDE_INDEX>(),
			std::move(cb)));
}

int trap_attach_impl::attach_at_with_ebpf_callback(void *func_addr,
						   ebpf_callback_args &&cb)
{
	return attach_at(func_addr, attach_entry_callback(std::move(cb)));
}

void trap_attach_impl::iterate_attaches(attach_iterate_callback cb)
{
	for (const auto &[id, e] : attaches)
		cb(id, (const void *)e.function, e.type);
}

int trap_attach_impl::detach_by_func_addr(const void *func)
{
	std::vector<int> ids;
	int res = trap_engine::get().detach_all_at((uintptr_t)func, this, ids);
	for (int id : ids)
		attaches.erase(id);
	return res;
}

int trap_attach_impl::detach_by_id(int id)
{
	auto it = attaches.find(id);
	if (it == attaches.end()) {
		SPDLOG_ERROR("Unable to find attach id {}", id);
		return -ENOENT;
	}
	int res = trap_engine::get().detach(id, this);
	attaches.erase(it);
	return res;
}

extern "C" uint64_t bpftime_set_retval(uint64_t value);

int trap_attach_impl::create_attach_with_ebpf_callback(
	ebpf_run_callback &&cb, const attach_private_data &private_data,
	int attach_type)
{
	SPDLOG_DEBUG("Attaching with private_data type {}",
		     typeid(private_data).name());
	const auto *sub =
		dynamic_cast<const trap_attach_private_data *>(&private_data);
	if (!sub) {
		SPDLOG_ERROR(
			"Trap attach impl expected a private data of type trap_attach_private_data");
		return -EINVAL;
	}
	SPDLOG_DEBUG(
		"Attaching with ebpf callback, private data addr={:x}, module name={}",
		sub->addr, sub->module_name);
	if (!sub->module_name.empty()) {
		bool ok = false;
		std::ifstream ifs("/proc/self/maps");
		std::string line;
		while (std::getline(ifs, line)) {
			char *module_path;
			if (sscanf(line.c_str(), "%*s%*s%*s%*s%*s%ms",
				   &module_path) != 1)
				continue;
			std::string curr_module(module_path);
			free(module_path);
			std::error_code ec;
			if (std::filesystem::exists(curr_module, ec) &&
			    std::filesystem::equivalent(sub->module_name,
							curr_module, ec)) {
				ok = true;
				break;
			}
		}
		if (!ok) {
			SPDLOG_INFO(
				"Unable to attach: module name {} doesn't exist in current process's memory maps",
				sub->module_name);
			return -EINVAL;
		}
	}
	void *func = (void *)(uintptr_t)sub->addr;
	if (attach_type == ATTACH_UPROBE || attach_type == ATTACH_URETPROBE ||
	    attach_type == ATTACH_UPROBE_OVERRIDE) {
		return attach_at_with_ebpf_callback(
			func, ebpf_callback_args{ .ebpf_cb = std::move(cb),
						  .attach_type = attach_type });
	} else if (attach_type == ATTACH_UREPLACE) {
		return attach_at_with_ebpf_callback(
			func,
			ebpf_callback_args{
				.ebpf_cb =
					[cb = std::move(cb)](
						void *memory, size_t memory_size,
						uint64_t *return_value) -> int {
					int err = cb(memory, memory_size,
						     return_value);
					if (err < 0) {
						SPDLOG_ERROR(
							"Failed to run ebpf callback at trap attach impl for ureplace, err={}",
							err);
						return err;
					}
					bpftime_set_retval(*return_value);
					return err;
				},
				.attach_type = ATTACH_UPROBE_OVERRIDE });
	}
	SPDLOG_ERROR("Unsupported attach type by trap attach impl: {}",
		     attach_type);
	return -ENOTSUP;
}

static constexpr int BPF_FUNC_get_func_arg = 183;
static constexpr int BPF_FUNC_get_func_ret = 184;
static constexpr int BPF_FUNC_get_retval = 186;

void trap_attach_impl::register_custom_helpers(
	ebpf_helper_register_callback register_callback)
{
	register_callback(BPF_FUNC_get_func_arg, "bpf_get_func_arg",
			  (void *)bpftime_trap_get_func_arg);
	register_callback(BPF_FUNC_get_func_ret, "bpf_get_func_ret_id",
			  (void *)bpftime_trap_get_func_ret);
	register_callback(BPF_FUNC_get_retval, "bpf_get_retval",
			  (void *)bpftime_trap_get_retval);
}

void *trap_attach_impl::call_attach_specific_function(const std::string &name,
						      void *)
{
	if (name == "generate_stack")
		return trap_engine::get().generate_stack();
	SPDLOG_ERROR("Invalid trap attach impl feature: {}", name);
	return nullptr;
}

extern "C" uint64_t
bpftime::attach::trap::bpftime_trap_get_func_arg(uint64_t, uint32_t n,
						 uint64_t *value, uint64_t,
						 uint64_t)
{
	if (tl_current_regs == nullptr)
		return -EINVAL;
	if (!arch::get_arg(*tl_current_regs, n, value))
		return -EINVAL;
	return 0;
}

extern "C" uint64_t
bpftime::attach::trap::bpftime_trap_get_func_ret(uint64_t, uint64_t *value,
						 uint64_t, uint64_t, uint64_t)
{
	if (tl_phase != 2)
		return -EOPNOTSUPP;
	*value = tl_return_value;
	return 0;
}

extern "C" uint64_t bpftime::attach::trap::bpftime_trap_get_retval(uint64_t,
								    uint64_t,
								    uint64_t,
								    uint64_t,
								    uint64_t)
{
	if (tl_phase != 2)
		return -EOPNOTSUPP;
	return tl_return_value;
}
