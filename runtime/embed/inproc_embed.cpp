// SPDX-License-Identifier: Apache-2.0
//
// bpftime in-process no-frida in-process embed. The LD_PRELOAD alloc shim calls the
// C ABI below; instead of frida rewriting the trampoline entry points, the
// attached eBPF program is executed synchronously in the normal call context
// with a synthesized kernel-style pt_regs. No GumInterceptor / boost::fiber
// path is ever created, which removes the long-run fiber SIGSEGV surface.
#include "bpftime_inproc_embed.h"

#include "attach_private_data.hpp"
#include "base_attach_impl.hpp"
#include "bpf_attach_ctx.hpp"
#include "bpftime_config.hpp"
#include "bpftime_shm.hpp"
#include "spdlog/cfg/env.h"
#include "spdlog/spdlog.h"

#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

// Attach type integers shared with the uprobe perf-event handlers. Defined
// locally so this target does not depend on the frida attach headers.
#define BPFTIME_INPROC_ATTACH_UPROBE 6
#define BPFTIME_INPROC_ATTACH_URETPROBE 7
#define BPFTIME_INPROC_ATTACH_KPROBE 8
#define BPFTIME_INPROC_ATTACH_KRETPROBE 9
#define BPFTIME_INPROC_ATTACH_TRACEPOINT 2
#define BPFTIME_INPROC_ATTACH_UPROBE_OVERRIDE 1008
#define BPFTIME_INPROC_ATTACH_UREPLACE 1009
#define BPFTIME_INPROC_MAX_TRAMPOLINES 8

// Kernel-only perf sources cannot be attached in rootless userspace. They are
// accepted as no-ops so handler instantiation does not abort the whole context.
static bool is_kernel_only_type(int t)
{
	return t == BPFTIME_INPROC_ATTACH_TRACEPOINT || t == BPFTIME_INPROC_ATTACH_KPROBE ||
	       t == BPFTIME_INPROC_ATTACH_KRETPROBE;
}
#if defined(__x86_64__) || defined(_M_X64)
struct inproc_pt_regs {
	uint64_t r15, r14, r13, r12, bp, bx, r11, r10, r9, r8, ax, cx, dx, si,
		di, orig_ax, ip, cs, flags, sp, ss;
};
#elif defined(__aarch64__) || defined(_M_ARM64)
struct inproc_pt_regs {
	uint64_t regs[31];
	uint64_t sp;
	uint64_t pc;
	uint64_t pstate;
};
#else
#error "in-process embed: unsupported architecture"
#endif

using namespace bpftime;
using namespace bpftime::attach;

// Lightweight stderr-only diagnostics. No file I/O: writing a FUSE mount from a
// forked Ray worker is unsafe, so diagnostics never touch the filesystem.
static void inproc_dbg(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
}
namespace
{
struct inproc_private_data : attach_private_data {
	std::string raw;
	bool has_colon = false;
	std::string module;
	uintptr_t address = 0;
	uintptr_t offset = 0;
	int initialize_from_string(const std::string_view &sv) override
	{
		raw = std::string(sv);
		auto pos = raw.find_last_of(':');
		try {
			if (pos == std::string::npos) {
				address = std::stoull(raw);
			} else {
				has_colon = true;
				module = raw.substr(0, pos);
				auto slash = module.find_last_of('/');
				if (slash != std::string::npos)
					module = module.substr(slash + 1);
				offset = std::stoull(raw.substr(pos + 1));
			}
		} catch (...) {
			// Non-address specs (e.g. kprobe func names like
			// __x64_sys_mmap) carry no module:offset. Keep the raw text and
			// succeed: kernel-only attaches no-op in create_attach, so a
			// parse miss here must not abort context initialization.
		}
		return 0;
	}
	std::string to_string() const override { return raw; }
};

struct slot {
	bool used = false;
	ebpf_run_callback cb;
};

// One impl owns all in-process uprobe attaches; callbacks are indexed by the stable
// trampoline id so the shim can trigger them without any symbol lookup.
class inproc_attach_impl final : public base_attach_impl
{
      public:
	std::array<slot, BPFTIME_INPROC_MAX_TRAMPOLINES> slots;
	std::array<std::pair<std::string, uintptr_t>, BPFTIME_INPROC_MAX_TRAMPOLINES>
		symbols{};
	std::array<uintptr_t, BPFTIME_INPROC_MAX_TRAMPOLINES> symbol_addrs{};
	std::array<uintptr_t, BPFTIME_INPROC_MAX_TRAMPOLINES> module_bases{};

	int create_attach_with_ebpf_callback(
		ebpf_run_callback &&cb, const attach_private_data &pd,
		int attach_type) override
	{
		int id = allocate_id();
		// Rootless cannot service kernel probes/tracepoints: accept the
		// attach as a no-op so context init proceeds.
		if (is_kernel_only_type(attach_type)) {
			(void)cb;
			(void)pd;
			return id;
		}
	const auto &d = dynamic_cast<const inproc_private_data &>(pd);
	inproc_dbg(
		     "BPFTIME_INPROC_EMBED_DBG create type=%d raw=%s colon=%d mod=%s off=%p addr=%p\n",
		     attach_type, d.raw.c_str(), d.has_colon ? 1 : 0,
		     d.module.c_str(), reinterpret_cast<void *>(d.offset),
		     reinterpret_cast<void *>(d.address));
	int idx = resolve_slot(d);
	if (idx < 0) {
		inproc_dbg(
			     "BPFTIME_INPROC_EMBED_DBG unresolved uprobe raw=%s -> ignored\n",
			     d.raw.c_str());
		return id; // do not fail init for unrelated uprobes
	}
	slots[idx].used = true;
	slots[idx].cb = std::move(cb);
	inproc_dbg( "BPFTIME_INPROC_EMBED_DBG bound trampoline=%d raw=%s\n",
		     idx + 1, d.raw.c_str());
	return id;
	}
	int detach_by_id(int id) override { return id >= 0 ? 0 : -EINVAL; }

	int run(int trampoline, const bpftime_inproc_embed_ctx &e)
	{
		static thread_local int dbg_n = 0;
		bool dbg = dbg_n++ < 6;
		if (trampoline < 1 || trampoline > BPFTIME_INPROC_MAX_TRAMPOLINES)
			return -EINVAL;
		auto &s = slots[trampoline - 1];
		if (!s.used || !s.cb)
			return -ENOENT;
		inproc_pt_regs regs;
		std::memset(&regs, 0, sizeof(regs));
#if defined(__x86_64__) || defined(_M_X64)
		regs.di = e.arg1;
		regs.si = e.arg2;
		regs.dx = e.arg3;
		regs.cx = e.arg4;
		regs.bp = e.frame_bp;
		regs.ip = e.probe_ip;
		regs.sp = e.frame_bp;
#else
		regs.regs[0] = e.arg1;
		regs.regs[1] = e.arg2;
		regs.regs[2] = e.arg3;
		regs.regs[3] = e.arg4;
		regs.regs[29] = e.frame_bp;
		regs.regs[30] = e.caller_ip;
		regs.pc = e.probe_ip;
		regs.sp = e.frame_bp;
#endif
	uint64_t ret = 0;
	int rc = s.cb(&regs, sizeof(regs), &ret);
	if (dbg)
		inproc_dbg(
			     "BPFTIME_INPROC_EMBED_DBG trigger tramp=%d arg1=%p arg2=%p rc=%d ret=%llu\n",
			     trampoline, reinterpret_cast<void *>(e.arg1),
			     reinterpret_cast<void *>(e.arg2), rc,
			     (unsigned long long)ret);
	const_cast<bpftime_inproc_embed_ctx &>(e).result_out = ret;
	return rc;
	}

      private:
	int resolve_slot(const inproc_private_data &d)
	{
		auto base_of = [](const std::string &p) -> std::string {
			auto slash = p.find_last_of('/');
			return slash == std::string::npos ? p :
							     p.substr(slash + 1);
		};
		std::string target_base = base_of(d.module);
		for (int i = 0; i < 3; ++i) {
			if (!symbol_addrs[i])
				continue;
			// Path A: handler gave a bare absolute address.
			if (!d.has_colon && d.address &&
			    d.address == symbol_addrs[i])
				return i;
			if (d.has_colon) {
				bool same_module =
					!target_base.empty() &&
					target_base == symbols[i].first;
				// Relative offset is position-independent and
				// survives PIE/ASLR base differences; accept a
				// match on offset alone within the same basename,
				// or when only one shim module is present.
				bool same_offset = d.offset == symbols[i].second;
				bool same_abs =
					module_bases[i] + d.offset ==
					symbol_addrs[i];
				if (same_abs || (same_module && same_offset))
					return i;
			}
		}
		// Last resort: a single embedded shim means every module:offset
		// uprobe targets one of our three trampolines; match by relative
		// offset regardless of how the module path was spelled.
		if (d.has_colon && d.offset) {
			for (int i = 0; i < 3; ++i)
				if (symbol_addrs[i] &&
				    d.offset == symbols[i].second)
					return i;
		}
		// Diagnose the mismatch against every known symbol.
		for (int i = 0; i < 3; ++i) {
			if (!symbol_addrs[i])
				continue;
			inproc_dbg(
				     "BPFTIME_INPROC_EMBED_DBG   cmp i=%d mod=%s off=%p abs=%p | target mod=%s off=%p addr=%p\n",
				     i, symbols[i].first.c_str(),
				     reinterpret_cast<void *>(symbols[i].second),
				     reinterpret_cast<void *>(symbol_addrs[i]),
				     d.module.c_str(),
				     reinterpret_cast<void *>(d.offset),
				     reinterpret_cast<void *>(d.address));
		}
		return -1;
	}
	uintptr_t symbols_addr(int i) const
	{
		Dl_info info;
		void *p = dlsym(RTLD_DEFAULT, symbol_name(i));
		if (p && dladdr(p, &info))
			return reinterpret_cast<uintptr_t>(p);
		return 0;
	}
	static const char *symbol_name(int i)
	{
		static const char *names[BPFTIME_INPROC_MAX_TRAMPOLINES] = {
			"inproc_sample_alloc",
			"inproc_sample_free",
			"inproc_pid_sync",
		};
		return i < 3 ? names[i] : nullptr;
	}
};

std::mutex g_init_mutex;
std::optional<bpf_attach_ctx> g_ctx;
inproc_attach_impl *g_impl = nullptr;
bool g_initialized = false;
int g_open_error = 0;

void build_symbols(inproc_attach_impl &impl)
{
	static const char *names[3] = { "inproc_sample_alloc",
					"inproc_sample_free",
					"inproc_pid_sync" };
	for (int i = 0; i < 3; ++i) {
		void *p = dlsym(RTLD_DEFAULT, names[i]);
		Dl_info info;
		inproc_dbg("BPFTIME_INPROC_EMBED_DBG build[%d] name=%s dlsym=%p\n", i,
			 names[i], p);
		if (p && dladdr(p, &info) && info.dli_fbase) {
			auto base = reinterpret_cast<uintptr_t>(info.dli_fbase);
			auto addr = reinterpret_cast<uintptr_t>(p);
			const char *fn = info.dli_fname ? info.dli_fname : "";
			auto slash = std::string(fn).find_last_of('/');
			std::string base_name =
				slash == std::string::npos ? fn :
							     std::string(fn)
								     .substr(slash + 1);
			impl.symbols[i] = { base_name, addr - base };
			impl.symbol_addrs[i] = addr;
			impl.module_bases[i] = base;
			inproc_dbg(
				"BPFTIME_INPROC_EMBED_DBG build[%d] mod=%s base=%p off=%p\n",
				i, base_name.c_str(),
				reinterpret_cast<void *>(base),
				reinterpret_cast<void *>(addr - base));
		}
	}
}
} // namespace

extern "C" int bpftime_inproc_embed_abi_version(void)
{
	return BPFTIME_INPROC_EMBED_ABI_VERSION;
}

extern "C" int bpftime_inproc_embed_open(const char * /*shm_name*/,
			       const char * /*config_path*/)
{
	std::lock_guard<std::mutex> guard(g_init_mutex);
	if (g_initialized)
		return g_open_error;
	try {
		spdlog::cfg::load_env_levels();
		bpftime_initialize_global_shm(
			shm_open_type::SHM_CREATE_OR_OPEN);
		g_ctx.emplace();
		auto impl = std::make_unique<inproc_attach_impl>();
		g_impl = impl.get();
		build_symbols(*g_impl);
		// Uprobe types bind real callbacks; kernel-only types are
		// registered so they no-op instead of aborting context init.
		g_ctx->register_attach_impl(
			{ BPFTIME_INPROC_ATTACH_TRACEPOINT, BPFTIME_INPROC_ATTACH_KPROBE,
			  BPFTIME_INPROC_ATTACH_KRETPROBE, BPFTIME_INPROC_ATTACH_UPROBE,
			  BPFTIME_INPROC_ATTACH_URETPROBE, BPFTIME_INPROC_ATTACH_UPROBE_OVERRIDE,
			  BPFTIME_INPROC_ATTACH_UREPLACE },
			std::move(impl),
			[](const std::string_view sv, int &err)
				-> std::unique_ptr<attach_private_data> {
				auto pd =
					std::make_unique<inproc_private_data>();
				err = pd->initialize_from_string(sv);
				return err == 0 ?
					       std::unique_ptr<
						       attach_private_data>(
						       std::move(pd)) :
						       nullptr;
			});
		int rc = g_ctx->init_attach_ctx_from_handlers(
			bpftime_get_runtime_config());
		// Kernel-only probes are expected to be skipped in rootless; their
		// per-handler rc must not block the uprobe path. Once the context is
		// constructed and the impl is registered, we are ready to trigger.
		if (rc != 0)
			SPDLOG_WARN(
				"in-process embed: init rc={} (kernel-only probes ignored)",
				rc);
		g_open_error = 0;
		g_initialized = true;
	} catch (const std::exception &ex) {
		// Likely shared memory/handlers not ready yet at shim constructor
		// time; leave g_initialized false so the next trigger retries.
		SPDLOG_WARN("in-process embed init deferred: {}", ex.what());
		g_open_error = -1;
		g_ctx.reset();
		g_impl = nullptr;
	}
	return g_initialized ? 0 : g_open_error;
}

extern "C" int bpftime_inproc_embed_trigger(const bpftime_inproc_embed_ctx *ctx)
{
	if (!ctx)
		return -EINVAL;
	// Lazy (re)initialization: at shim constructor time the collector may not
	// have published handlers yet, so retry on every trigger until ready.
	if (!g_initialized) {
		int rc = bpftime_inproc_embed_open(nullptr, nullptr);
		if (rc != 0 || !g_impl)
			return -ENXIO;
	}
	try {
		return g_impl->run(ctx->trampoline, *ctx);
	} catch (const std::exception &ex) {
		SPDLOG_DEBUG("in-process embed trigger failed: {}", ex.what());
		return -EFAULT;
	}
}

extern "C" void bpftime_inproc_embed_close(void)
{
	// Shared memory and programs outlive the shim and are torn down with
	// the process group; nothing to release per process here.
}
