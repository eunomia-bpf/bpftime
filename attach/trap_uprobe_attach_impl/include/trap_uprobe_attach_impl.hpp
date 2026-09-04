/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
// Trap based uprobe / uretprobe attach implementation.
//
// Instead of rewriting the function prologue with a jump (what frida-gum
// does), this backend writes a breakpoint instruction at the probe address
// and services the resulting SIGTRAP. The original instruction is either
// executed from an out-of-line slot or emulated, so probes never lose events
// and never require a free register. The price is two signal deliveries per
// hit, which makes this the portable but slower backend: it is the default
// on architectures without frida support (such as riscv64) and can be
// selected explicitly elsewhere with BPFTIME_UPROBE_BACKEND=trap.
#ifndef _BPFTIME_TRAP_UPROBE_ATTACH_IMPL_HPP
#define _BPFTIME_TRAP_UPROBE_ATTACH_IMPL_HPP

#include "base_attach_impl.hpp"
#include "bpftime_pt_regs.hpp"
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace bpftime
{
namespace attach
{
namespace trap
{
// Attach type ids. They are the same numbers the frida backend uses, so the
// runtime can route a link to either backend transparently.
constexpr int ATTACH_UPROBE = 6;
constexpr int ATTACH_URETPROBE = 7;
constexpr int ATTACH_UPROBE_OVERRIDE = 1008;
constexpr int ATTACH_UREPLACE = 1009;

constexpr int ATTACH_UPROBE_INDEX = 0;
constexpr int ATTACH_URETPROBE_INDEX = 1;
constexpr int ATTACH_UPROBE_OVERRIDE_INDEX = 2;

using uprobe_callback = std::function<void(const pt_regs &regs)>;
using uretprobe_callback = std::function<void(const pt_regs &regs)>;
using uprobe_override_callback = std::function<void(const pt_regs &regs)>;
using callback_variant = std::variant<uprobe_callback, uretprobe_callback,
				      uprobe_override_callback>;
using attach_iterate_callback =
	std::function<void(int id, const void *addr, int ty)>;

struct ebpf_callback_args {
	ebpf_run_callback ebpf_cb;
	int attach_type;
};

using attach_entry_callback = std::variant<callback_variant, ebpf_callback_args>;

int from_cb_idx_to_attach_type(int idx);

// Check whether a probe can be placed at `func_addr`. Returns a diagnostic
// message when the first instruction cannot be relocated, std::nullopt when
// the address is usable.
std::optional<std::string> check_probe_target(const void *func_addr);

class trap_attach_impl final : public base_attach_impl {
    public:
	trap_attach_impl();
	~trap_attach_impl() override;

	// Pre-allocate per-thread state (uretprobe shadow stack) so that the
	// SIGTRAP handler never needs to call mmap.  Call this once from each
	// thread that will hit probes, before the first hit.  It is safe but
	// unnecessary to call more than once.
	static void prepare_thread();
	trap_attach_impl(const trap_attach_impl &) = delete;
	trap_attach_impl &operator=(const trap_attach_impl &) = delete;

	int create_uprobe_at(void *func_addr, uprobe_callback &&cb);
	int create_uretprobe_at(void *func_addr, uretprobe_callback &&cb);
	int create_uprobe_override_at(void *func_addr,
				      uprobe_override_callback &&cb);
	int attach_at_with_ebpf_callback(void *func_addr,
					 ebpf_callback_args &&cb);
	void iterate_attaches(attach_iterate_callback cb);
	int detach_by_func_addr(const void *func);

	int detach_by_id(int id) override;
	int create_attach_with_ebpf_callback(
		ebpf_run_callback &&cb, const attach_private_data &private_data,
		int attach_type) override;
	void register_custom_helpers(
		ebpf_helper_register_callback register_callback) override;
	void *call_attach_specific_function(const std::string &name,
					    void *data) override;

    private:
	int attach_at(void *func_addr, attach_entry_callback &&cb);
	// id -> function address / attach type of entries owned by this
	// instance. The entries themselves live in the shared engine.
	struct owned_entry {
		uintptr_t function;
		int type;
	};
	std::unordered_map<int, owned_entry> attaches;
};

// Helper implementations registered through register_custom_helpers
extern "C" uint64_t bpftime_trap_get_func_arg(uint64_t ctx, uint32_t n,
					      uint64_t *value, uint64_t,
					      uint64_t);
extern "C" uint64_t bpftime_trap_get_func_ret(uint64_t ctx, uint64_t *value,
					      uint64_t, uint64_t, uint64_t);
extern "C" uint64_t bpftime_trap_get_retval(uint64_t, uint64_t, uint64_t,
					    uint64_t, uint64_t);
} // namespace trap
} // namespace attach
} // namespace bpftime

#endif
