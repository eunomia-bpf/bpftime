/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "trap_uprobe_attach_impl.hpp"
#include "trap_attach_private_data.hpp"
#include "trap_attach_utils.hpp"
#include "trap_arch.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <span>
#include <spdlog/spdlog.h>
#include <sys/mman.h>
#include <sys/syscall.h>
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
	std::atomic<bool> enabled{ true };
	mutable std::atomic<unsigned> active_callbacks{ 0 };

	template <int callback_index> void run(const pt_regs &regs) const
	{
		struct callback_scope {
			std::atomic<unsigned> &active;
			~callback_scope() { active.fetch_sub(1); }
		};
		active_callbacks.fetch_add(1);
		callback_scope scope{ active_callbacks };
		if (!enabled.load())
			return;
		current_signal_callback->function_ip =
			(callback_index == ATTACH_UPROBE_INDEX ||
			 callback_index == ATTACH_URETPROBE_INDEX) ? function : 0;
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
struct page_protection {
	uintptr_t addr;
	size_t len;
	int prot;
};

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
	bool protections_dirty = false;
	std::vector<page_protection> protections;
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

// Only two small pointers (this one and current_signal_callback) require
// static TLS. No dynamic initialization, TLS destructor, allocator or lock
// runs on a thread's first hit, including threads predating dlopen.
struct trap_tls {
	std::atomic<bool> owned{ false };
	int in_handler = 0;
	uret_stack uret{};
	const pt_regs *current_regs = nullptr;
	uintptr_t current_pc = 0;
	uintptr_t current_sp = 0;
	int phase = 0;
	uint64_t return_value = 0;
	override_state override{};
	signal_callback_context callback{};
};
constexpr size_t THREAD_SLOTS = 1024;
constinit trap_tls thread_slots[THREAD_SLOTS];
static_assert(std::atomic<bool>::is_always_lock_free);
static_assert(std::atomic<int>::is_always_lock_free);
static_assert(std::atomic<unsigned>::is_always_lock_free);
static_assert(std::atomic<uint64_t>::is_always_lock_free);
static_assert(std::atomic<void *>::is_always_lock_free);
__attribute__((tls_model("initial-exec"))) thread_local trap_tls *tl_slot = nullptr;
std::atomic<uint64_t> exhausted_slots{ 0 };

trap_tls *claim_thread_slot()
{
	if (tl_slot)
		return tl_slot;
	for (auto &slot : thread_slots) {
		bool expected = false;
		if (slot.owned.compare_exchange_strong(expected, true,
			std::memory_order_acquire, std::memory_order_relaxed)) {
			tl_slot = &slot;
			return tl_slot;
		}
	}
	exhausted_slots.fetch_add(1, std::memory_order_relaxed);
	return nullptr;
}

void release_idle_slot()
{
	if (tl_slot && !tl_slot->in_handler && tl_slot->uret.depth == 0) {
		auto *slot = tl_slot;
		tl_slot = nullptr;
		slot->owned.store(false, std::memory_order_release);
	}
}

void set_override(uint64_t ctx, uint64_t value)
{
	if (tl_slot) {
		tl_slot->override.is_overrided = true;
		tl_slot->override.value = value;
		tl_slot->override.ctx = ctx;
	}
}

std::atomic<uint64_t> split_writes{ 0 };

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

// Outcome of a code patch. Callers need to tell apart "nothing happened" from
// "the bytes changed but the page could not be locked down again", because the
// two require different recovery.
enum class patch_result {
	// Bytes written and the page is back to its original protection.
	ok,
	// The page could not be made writable; the site is untouched.
	failed_before_write,
	// The bytes were written but the original protection could not be
	// restored, so the page may still be writable. The caller must put the
	// site back into a defined state and report the failure.
	failed_after_write,
};

// Read protections outside the handler. Unknown mappings must fail closed;
// guessing RX can silently change the host's original permissions.
std::optional<int> read_page_prot(uintptr_t addr)
{
	FILE *maps = fopen("/proc/self/maps", "re");
	if (!maps)
		return std::nullopt;
	std::optional<int> result;
	char *line = nullptr;
	size_t cap = 0;
	while (getline(&line, &cap, maps) > 0) {
		unsigned long lo, hi;
		char perms[8] = {};
		if (sscanf(line, "%lx-%lx %4s", &lo, &hi, perms) != 3 ||
		    addr < lo || addr >= hi)
			continue;
		result = (perms[0] == 'r' ? PROT_READ : 0) |
			 (perms[1] == 'w' ? PROT_WRITE : 0) |
			 (perms[2] == 'x' ? PROT_EXEC : 0);
		break;
	}
	free(line);
	fclose(maps);
	return result;
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
	int detach(int id, trap_attach_impl *owner, bool &detached);
	int detach_all_at(uintptr_t function, trap_attach_impl *owner,
			  std::vector<int> &removed_ids);
	void detach_owner(trap_attach_impl *owner);

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
	patch_result write_code(const probe_site &site, const uint8_t *data);
	// Store the patch without touching page protection. Only valid while
	// the page is known to be writable; used to undo a half-applied patch.
	void store_patch(uintptr_t addr, const uint8_t *data, size_t len);
	// RCU-style swap: atomically replace the site_table pointer so that
	// in-flight signal handlers see either the old or the new table.
	void publish(std::unique_ptr<site_table> table);
	// Disable removed entries and wait for their active callbacks before
	// allowing the caller to destroy callback-owned state.
	void wait_for_quiescence(std::span<attach_entry *const> removed);
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

// Allocate a SLOT_SIZE-byte out-of-line execution slot near `near` (within
// ±1 GB so that PC-relative references still reach).  Reuses existing
// regions first, then tries mmap with RWX, then falls back to W^X.
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

// Store `len` bytes of code at `addr` assuming the page is already writable.
// The store is done with a single aligned write of the natural size so that
// other threads observe either the old or the new instruction.  For naturally
// aligned 2- and 4-byte writes a single atomic store suffices; a 4-byte write
// at a 2-byte boundary uses the three-phase c.ebreak protocol so that a
// concurrent instruction fetch always sees either the old instruction or a
// complete trap—never a torn mix of old and new.
void trap_engine::store_patch(uintptr_t addr, const uint8_t *data, size_t len)
{
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
		split_writes.fetch_add(1, std::memory_order_relaxed);
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
}

// Never remove EXEC from live host code as a fallback for W^X rejection:
// another thread can be executing any instruction on the same page.
// Keep the originally captured permissions across failed operations so that
// a retry cannot mistake a degraded RWX page for the host's intended state.
patch_result trap_engine::write_code(const probe_site &site, const uint8_t *data)
{
	size_t writable = 0;
	for (const auto &page : site.protections) {
		if (mprotect((void *)page.addr, page.len,
			     page.prot | PROT_WRITE) != 0) {
			for (size_t i = 0; i < writable; ++i) {
				const auto &prev = site.protections[i];
				if (mprotect((void *)prev.addr, prev.len, prev.prot) != 0)
					mprotect((void *)prev.addr, prev.len, prev.prot);
			}
			return patch_result::failed_before_write;
		}
		++writable;
	}
	store_patch(site.addr, data, site.trap_len);
	bool restored = true;
	for (const auto &page : site.protections) {
		if (mprotect((void *)page.addr, page.len, page.prot) != 0) {
			restored = false;
			// Best effort retry, but propagate even a transient failure.
			mprotect((void *)page.addr, page.len, page.prot);
		}
	}
	return restored ? patch_result::ok : patch_result::failed_after_write;
}

int trap_engine::build_site(uintptr_t addr, probe_site &site, std::string &err)
{
	site.addr = addr;
	auto info = arch::decode((const uint8_t *)addr, err);
	if (!info) {
		SPDLOG_ERROR("Cannot probe {:x}: {}", addr, err);
		return -ENOTSUP;
	}
	site.info = *info;
	std::memcpy(site.orig, (const void *)addr, site.info.len);
	site.trap_len = arch::trap_bytes(site.info.len, site.trap);
	const long page_size = sysconf(_SC_PAGESIZE);
	if (page_size <= 0)
		return -EIO;
	const uintptr_t page_mask = (uintptr_t)page_size - 1;
	for (uintptr_t page = addr & ~page_mask;
	     page <= (addr + site.trap_len - 1) ; page += page_size) {
		auto prot = read_page_prot(page);
		if (!prot) {
			err = "unable to determine original code page protections";
			return -EIO;
		}
		site.protections.push_back({ page, (size_t)page_size, *prot });
	}
	if (site.info.kind == arch::insn_kind::execute_out_of_line) {
		bool writable;
		uint8_t *slot = alloc_slot(addr, writable);
		if (!slot) {
			err = "unable to allocate an out-of-line slot";
			SPDLOG_ERROR("Cannot probe {:x}: {}", addr, err);
			return -ENOMEM;
		}
		size_t trap_offset;
		// The slot is writable at this point: either the region is
		// mapped RWX, or it is a fresh RW page that finalize_slot
		// turns into RX below.
		if (!arch::prepare_out_of_line(site.orig, site.info, addr, slot,
					       SLOT_SIZE, &trap_offset, err)) {
			SPDLOG_ERROR("Cannot probe {:x}: {}", addr, err);
			return -ENOTSUP;
		}
		if (!finalize_slot(slot, writable, SLOT_SIZE)) {
			err = "unable to make the out-of-line slot executable";
			SPDLOG_ERROR("Cannot probe {:x}: {}", addr, err);
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

// A handler can have loaded an old site before publication. Disable its entry
// before waiting so that even a delayed reader cannot start another callback.
// Sequential consistency pairs the increment/enable check in run() with this
// disable/count check. Unrelated probes do not prevent this detach completing.
void trap_engine::wait_for_quiescence(std::span<attach_entry *const> removed)
{
	for (auto *entry : removed)
		entry->enabled.store(false);
	for (auto *entry : removed)
		while (entry->active_callbacks.load() != 0)
			sched_yield();
}

// Attach a probe at `function`.  Reuses an existing site when the address
// is already probed, otherwise decodes the instruction, allocates an
// out-of-line slot (if the instruction can be relocated), publishes the
// new site table, and then writes the trap instruction.  The trap is
// written *after* publishing so that a concurrent hit always finds its
// site in the table.
int trap_engine::attach(trap_attach_impl *owner, int id, uintptr_t function,
			attach_entry_callback &&cb, int type)
{
	std::lock_guard<std::mutex> guard(mu);
	if (!ensure_installed()) {
		SPDLOG_ERROR("Failed to install SIGTRAP handler");
		return -EIO;
	}
	site_table *cur = table.load(std::memory_order_acquire);
	probe_site *existing = cur ? cur->find_by_addr(function) : nullptr;
	if (existing && existing->protections_dirty) {
		for (const auto &page : existing->protections)
			if (mprotect((void *)page.addr, page.len, page.prot) != 0)
				return -EIO;
	}
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
	site->protections_dirty = false;

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
		auto patched = write_code(*site_ptr, site_ptr->trap);
		if (patched != patch_result::ok) {
			// Reacquire write permission before undoing: restoration may
			// have succeeded on a retry or on only one of two pages.
			auto rollback = patch_result::ok;
			if (patched == patch_result::failed_after_write)
				rollback = write_code(*site_ptr, site_ptr->orig);
			auto disabled = std::make_unique<probe_site>(*site_ptr);
			disabled->entries.clear();
			disabled->has_override = disabled->has_uprobe =
				disabled->has_uretprobe = false;
			// If rollback itself cannot write, retain a callback-free
			// trap site so any hit still executes the original instruction.
			disabled->armed = rollback == patch_result::failed_before_write;
			disabled->protections_dirty = true;
			publish(copy_table_with(disabled.get()));
			sites.push_back(std::move(disabled));
			wait_for_quiescence(site_ptr->entries);
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

int trap_engine::detach(int id, trap_attach_impl *owner, bool &detached)
{
	detached = false;
	std::lock_guard<std::mutex> guard(mu);
	site_table *cur = table.load(std::memory_order_acquire);
	if (!cur) {
		SPDLOG_WARN("detach id {}: no probe table", id);
		return -ENOENT;
	}
	for (auto *s : cur->by_addr) {
		for (auto *e : s->entries) {
			if (e->id != id || e->owner != owner)
				continue;
			auto site = std::make_unique<probe_site>(*s);
			remove_entries_from_site(*site, { e });
			// The original instruction is always restored; only the
			// page protection can be left degraded, which is
			// reported after the detach has been completed so the
			// probe table never keeps a stale entry.
			auto patched = patch_result::ok;
			if (site->entries.empty()) {
				patched = write_code(*site, site->orig);
				if (patched == patch_result::failed_before_write) {
					// Preserve the entry for retry, including the original
					// protections if a partial permission rollback failed.
					auto retry = std::make_unique<probe_site>(*s);
					retry->protections_dirty = true;
					publish(copy_table_with(retry.get()));
					sites.push_back(std::move(retry));
					return -EIO;
				}
				site->armed = false;
				site->protections_dirty = patched != patch_result::ok;
				SPDLOG_DEBUG("Disarmed trap uprobe at {:x}",
					     site->addr);
			}
			detached = true;
			publish(copy_table_with(site.get()));
			sites.push_back(std::move(site));
			wait_for_quiescence({ &e, 1 });
			if (patched != patch_result::ok) {
				SPDLOG_ERROR(
					"Disarmed trap uprobe at {:x} but could not restore the page protection",
					s->addr);
				return -EIO;
			}
			return 0;
		}
	}
	SPDLOG_WARN("detach id {}: entry not found", id);
	return -ENOENT;
}

int trap_engine::detach_all_at(uintptr_t function, trap_attach_impl *owner,
			       std::vector<int> &removed_ids)
{
	std::lock_guard<std::mutex> guard(mu);
	site_table *cur = table.load(std::memory_order_acquire);
	probe_site *s = cur ? cur->find_by_addr(function) : nullptr;
	if (!s || !s->armed) {
		SPDLOG_WARN("detach_all_at {:x}: site not found or not armed",
			    function);
		return -ENOENT;
	}
	std::vector<attach_entry *> remove;
	for (auto *e : s->entries) {
		if (e->owner == owner) {
			remove.push_back(e);
			removed_ids.push_back(e->id);
		}
	}
	if (remove.empty()) {
		SPDLOG_WARN("detach_all_at {:x}: no entries owned by caller",
			    function);
		return -ENOENT;
	}
	auto site = std::make_unique<probe_site>(*s);
	remove_entries_from_site(*site, remove);
	auto patched = patch_result::ok;
	if (site->entries.empty()) {
		patched = write_code(*site, site->orig);
		site->armed = patched == patch_result::failed_before_write;
		site->protections_dirty = patched != patch_result::ok;
	}
	publish(copy_table_with(site.get()));
	sites.push_back(std::move(site));
	wait_for_quiescence(remove);
	if (patched != patch_result::ok) {
		SPDLOG_ERROR(
			"Disarmed trap uprobe at {:x} but could not restore the page protection",
			function);
		return -EIO;
	}
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

// Handle a breakpoint hit at a probed address.
// Override attaches short-circuit: the BPF program's return value replaces
// the function call.  Normal uprobes fire all callbacks, then install a
// uretprobe shadow-stack entry if any return probes are registered.
void trap_engine::handle_hit(probe_site *site, ucontext_t *uc)
{
	if (claim_thread_slot() && tl_slot->in_handler == 0 && site->armed && !site->entries.empty()) {
		tl_slot->in_handler = 1;
		current_signal_callback = &tl_slot->callback;
		tl_slot->callback.cookie.reset();
		pt_regs regs;
		arch::fill_pt_regs(uc, site->addr, regs);
		tl_slot->current_regs = &regs;
		tl_slot->current_pc = site->addr;
		tl_slot->current_sp = arch::get_sp(uc);
		tl_slot->phase = 1;
		bool overrided = false;
		try {
			if (site->has_override) {
				tl_slot->override = override_state{};
				tl_slot->callback.override_return = set_override;
				for (auto *e : site->entries) {
					if (e->type == ATTACH_UPROBE_OVERRIDE) {
						e->run<ATTACH_UPROBE_OVERRIDE_INDEX>(
							regs);
						break;
					}
				}
				tl_slot->callback.override_return = nullptr;
				overrided = tl_slot->override.is_overrided;
			} else {
				for (auto *e : site->entries) {
					if (e->type == ATTACH_UPROBE)
						e->run<ATTACH_UPROBE_INDEX>(regs);
				}
			}
		} catch (...) {
		}
		current_signal_callback = nullptr;
		tl_slot->callback.override_return = nullptr;
		tl_slot->phase = 0;
		tl_slot->current_regs = nullptr;
		if (overrided) {
			tl_slot->in_handler = 0;
			arch::do_return(uc, tl_slot->override.value);
			return;
		}
		if (site->has_uretprobe) {
			uret_stack *stack = &tl_slot->uret;
			if (stack->depth < URET_STACK_DEPTH) {
				auto &frame = stack->frames[stack->depth++];
				frame.function = site->addr;
				frame.orig_ret = arch::get_return_address(uc);
				frame.sp = arch::get_sp(uc);
				arch::set_return_address(uc, uret_trampoline);
			}
		}
		tl_slot->in_handler = 0;
	}
	resume(site, uc);
}

void trap_engine::handle_return(ucontext_t *uc)
{
	if (!tl_slot)
		return;
	uret_stack *stack = &tl_slot->uret;
	uintptr_t sp = arch::get_sp(uc);
	// Drop frames abandoned by longjmp/exceptions: their stack pointer is
	// below the one we are returning with.
	while (stack->depth > 0 &&
	       stack->frames[stack->depth - 1].sp + sizeof(uintptr_t) < sp)
		stack->depth--;
	if (stack->depth == 0) {
		return;
	}
	uret_frame frame = stack->frames[--stack->depth];
	arch::set_pc(uc, frame.orig_ret);
	if (tl_slot->in_handler != 0)
		return;
	site_table *cur = table.load(std::memory_order_acquire);
	probe_site *site = cur ? cur->find_by_addr(frame.function) : nullptr;
	if (!site || !site->armed || !site->has_uretprobe)
		return;
	tl_slot->in_handler = 1;
	current_signal_callback = &tl_slot->callback;
	tl_slot->callback.cookie.reset();
	pt_regs regs;
	arch::fill_pt_regs(uc, frame.orig_ret, regs);
	tl_slot->current_regs = &regs;
	tl_slot->current_pc = frame.orig_ret;
	tl_slot->current_sp = arch::get_sp(uc);
	tl_slot->phase = 2;
	tl_slot->return_value = arch::get_return_value(uc);
	try {
		for (auto *e : site->entries) {
			if (e->type == ATTACH_URETPROBE)
				e->run<ATTACH_URETPROBE_INDEX>(regs);
		}
	} catch (...) {
	}
	current_signal_callback = nullptr;
	tl_slot->phase = 0;
	tl_slot->current_regs = nullptr;
	tl_slot->in_handler = 0;
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

// SIGTRAP entry point.  Dispatch order:
//   1. uretprobe trampoline  → handle return-probe callbacks
//   2. out-of-line slot trap → resume after relocated instruction
//   3. known probe address   → run uprobe/override callbacks
//   4. none of the above     → forward to previous handler
void trap_engine::on_sigtrap(int sig, siginfo_t *info, void *ctx)
{
	int saved_errno = errno;
	++signal_handler_depth;
	auto *uc = (ucontext_t *)ctx;
	trap_engine &engine = trap_engine::get();
	uintptr_t pc = arch::trap_pc(uc);
	if (pc == engine.uret_trampoline && pc != 0) {
		engine.handle_return(uc);
	} else if (site_table *cur =
			   engine.table.load(std::memory_order_acquire);
		   cur != nullptr) {
		if (probe_site *s = cur->find_by_slot_trap(pc); s) {
			arch::set_pc(uc, s->addr + s->info.len);
		} else if (probe_site *hit = cur->find_by_addr(pc); hit) {
			engine.handle_hit(hit, uc);
		} else {
			engine.chain_previous(sig, info, ctx);
		}
	} else {
		engine.chain_previous(sig, info, ctx);
	}
	release_idle_slot();
	--signal_handler_depth;
	errno = saved_errno;
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

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

uint64_t bpftime::attach::trap::split_write_count()
{
	return split_writes.load(std::memory_order_relaxed);
}

uint64_t bpftime::attach::trap::thread_slot_exhaustion_count()
{
	return exhausted_slots.load(std::memory_order_relaxed);
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

void trap_attach_impl::prepare_thread()
{
	(void)claim_thread_slot();
	release_idle_slot();
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
	if (current_signal_callback)
		return -EOPNOTSUPP;
	const bool callable = std::holds_alternative<callback_variant>(cb) ?
		std::visit([](const auto &fn) { return bool(fn); },
			   std::get<callback_variant>(cb)) :
		bool(std::get<ebpf_callback_args>(cb).ebpf_cb);
	if (!callable)
		return -EINVAL;
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
	if (current_signal_callback)
		return -EOPNOTSUPP;
	std::vector<int> ids;
	int res = trap_engine::get().detach_all_at((uintptr_t)func, this, ids);
	for (int id : ids)
		attaches.erase(id);
	return res;
}

int trap_attach_impl::detach_by_id(int id)
{
	if (current_signal_callback)
		return -EOPNOTSUPP;
	auto it = attaches.find(id);
	if (it == attaches.end()) {
		SPDLOG_ERROR("Unable to find attach id {}", id);
		return -ENOENT;
	}
	bool detached = false;
	int res = trap_engine::get().detach(id, this, detached);
	if (detached)
		attaches.erase(it);
	return res;
}

extern "C" uint64_t bpftime_set_retval(uint64_t value);

int trap_attach_impl::create_attach_with_ebpf_callback(
	ebpf_run_callback &&cb, const attach_private_data &private_data,
	int attach_type)
{
	if (current_signal_callback)
		return -EOPNOTSUPP;
	if (!cb)
		return -EINVAL;
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
	if (!sub->module_name.empty() &&
	    get_module_base_addr(sub->module_name.c_str()) == nullptr) {
		SPDLOG_INFO(
			"Unable to attach: module {} is not loaded in this process",
			sub->module_name);
		return -EINVAL;
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

void *trap_attach_impl::call_attach_specific_function(const std::string &, void *)
{
	// Dynamic unwinding and heap-owned vectors are not signal-safe.
	return nullptr;
}

extern "C" uint64_t
bpftime::attach::trap::bpftime_trap_get_func_arg(uint64_t, uint32_t n,
						 uint64_t *value, uint64_t,
						 uint64_t)
{
	if (!tl_slot || tl_slot->current_regs == nullptr) {
		return -EINVAL;
	}
	if (!arch::get_arg(*tl_slot->current_regs, n, value)) {
		return -EINVAL;
	}
	return 0;
}

extern "C" uint64_t
bpftime::attach::trap::bpftime_trap_get_func_ret(uint64_t, uint64_t *value,
						 uint64_t, uint64_t, uint64_t)
{
	if (!tl_slot || tl_slot->phase != 2) {
		return -EOPNOTSUPP;
	}
	*value = tl_slot->return_value;
	return 0;
}

extern "C" uint64_t bpftime::attach::trap::bpftime_trap_get_retval(uint64_t,
								    uint64_t,
								    uint64_t,
								    uint64_t,
								    uint64_t)
{
	if (!tl_slot || tl_slot->phase != 2) {
		return -EOPNOTSUPP;
	}
	return tl_slot->return_value;
}
