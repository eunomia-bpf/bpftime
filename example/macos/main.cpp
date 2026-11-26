// SPDX-License-Identifier: MIT
// Cross-platform userspace eBPF uprobe/uretprobe demo driven through the
// bpftime in-process API: a shared-memory array map, embedded eBPF bytecode on
// the uBPF interpreter or JIT, and Frida-gum inline hooks on a function in a
// separate shared library. Runs unprivileged on macOS (Apple Silicon) and Linux.
#include "bpf_attach_ctx.hpp"
#include "bpftime_shm.hpp"
#include "frida_attach_private_data.hpp"
#include "frida_register_def.hpp"
#include "frida_uprobe_attach_impl.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unistd.h>

using namespace bpftime;
using namespace bpftime::attach;

extern "C" uint64_t bpftime_macos_demo_compute(uint64_t a, uint64_t b);

// One bytecode source for both architectures: only the pt_regs offsets differ,
// taken from the Frida register definition header.
#if defined(__aarch64__)
constexpr int PARM1_OFFSET = offsetof(pt_regs, regs[0]);
constexpr int RETURN_OFFSET = offsetof(pt_regs, regs[0]);
#elif defined(__x86_64__)
constexpr int PARM1_OFFSET = offsetof(pt_regs, di);
constexpr int RETURN_OFFSET = offsetof(pt_regs, ax);
#else
#error "macos_uprobe_demo supports arm64 and x86_64 only"
#endif

static void require(bool ok, const char *message)
{
	if (!ok)
		throw std::runtime_error(message);
}

static size_t open_fd_count()
{
#ifdef __APPLE__
	auto entries = std::filesystem::directory_iterator("/dev/fd");
#else
	auto entries = std::filesystem::directory_iterator("/proc/self/fd");
#endif
	return std::distance(begin(entries), end(entries));
}

static bool shm_residue_exists(const std::string &name)
{
#ifdef __APPLE__
	return std::filesystem::exists("/tmp/boost_interprocess/" + name);
#else
	return std::filesystem::exists("/dev/shm/" + name);
#endif
}

// Owns the whole runtime session; close() is idempotent and also runs from the
// destructor, so error paths release handlers, descriptors and shared memory.
struct session {
	bool owns_shm = false;
	int map_fd = -1, uprobe_prog_fd = -1, uret_prog_fd = -1,
	    uprobe_event_fd = -1, uret_event_fd = -1,
	    uprobe_link_fd = -1, uret_link_fd = -1;
	std::optional<bpf_attach_ctx> ctx;

	void close()
	{
		if (ctx) {
			ctx->destroy_all_attach_links();
			ctx->reset_instantiated_state();
			ctx.reset();
		}
		for (int fd : { uret_link_fd, uprobe_link_fd, uret_event_fd,
			       uprobe_event_fd, uret_prog_fd, uprobe_prog_fd,
			       map_fd })
			if (fd >= 0) {
				bpftime_close(fd);
				::close(fd);
			}
		uprobe_link_fd = uret_link_fd = uprobe_event_fd =
			uret_event_fd = uprobe_prog_fd = uret_prog_fd =
				map_fd = -1;
		if (owns_shm) {
			bpftime_destroy_global_shm();
			bpftime_remove_global_shm();
			owns_shm = false;
		}
	}
	~session() { close(); }
};

int main(int argc, char **argv)
{
	try {
		require(argc <= 3,
			"usage: macos_uprobe_demo [interpreter|jit] [iterations]");
		const std::string mode = argc > 1 ? argv[1] : "jit";
		require(mode == "interpreter" || mode == "jit",
			"mode must be interpreter or jit");
		const uint32_t iterations = argc > 2 ? std::stoul(argv[2]) : 100;
		require(iterations > 0, "iterations must be positive");
		spdlog::set_level(spdlog::level::err);

		const std::string name =
			"bpftime-macos-demo-" + std::to_string(getpid()) + "-" +
			std::to_string(std::chrono::steady_clock::now()
					       .time_since_epoch()
					       .count());
		require(setenv("BPFTIME_GLOBAL_SHM_NAME", name.c_str(), 1) == 0,
			"setting private SHM name failed");
		bpftime_remove_global_shm();
		bpftime_initialize_global_shm(shm_open_type::SHM_CREATE_ONLY);

		session s;
		s.owns_shm = true;
		const size_t fds_before = open_fd_count();

		runtime_config config{};
		config.set_vm_name("ubpf");
		config.jit_enabled = (mode == "jit");
		config.enable_kernel_helper_group = false;
		config.enable_ufunc_helper_group = false;
		config.enable_shm_maps_helper_group = true;
		bpftime_set_runtime_config(runtime_config(config));

		// Slots: 0 = call count, 1 = first-argument sum, 2 = return sum.
		bpf_map_attr attr{};
		attr.type = static_cast<uint32_t>(bpf_map_type::BPF_MAP_TYPE_ARRAY);
		attr.key_size = sizeof(uint32_t);
		attr.value_size = sizeof(uint64_t);
		attr.max_ents = 3;
		s.map_fd = bpftime_maps_create(-1, "macos_demo_stats", attr);
		require(s.map_fd >= 0, "map creation failed");
		for (uint32_t key = 0; key < 3; ++key) {
			const uint64_t zero = 0;
			require(bpftime_map_update_elem(s.map_fd, &key, &zero, 0) ==
					0,
				"map zero initialization failed");
		}
		// uprobe: increment the call counter (slot 0) and accumulate the
		// first argument read from the per-architecture pt_regs (slot 1).
		// Each group looks up one array slot and adds a value to it.
		const ebpf_inst uprobe_insns[] = {
			{ EBPF_OP_MOV64_REG, 6, 1, 0, 0 }, // r6 = ctx
			{ EBPF_OP_LDXDW, 7, 6, PARM1_OFFSET, 0 }, // r7 = arg1
			{ EBPF_OP_STW, 10, 0, -4, 0 }, // key = 0
			{ EBPF_OP_LDDW, 1, 1, 0, s.map_fd }, { 0, 0, 0, 0, 0 },
			{ EBPF_OP_MOV64_REG, 2, 10, 0, 0 },
			{ EBPF_OP_ADD64_IMM, 2, 0, 0, -4 },
			{ EBPF_OP_CALL, 0, 0, 0, 1 }, // bpf_map_lookup_elem
			{ EBPF_OP_JEQ_IMM, 0, 0, 3, 0 },
			{ EBPF_OP_LDXDW, 1, 0, 0, 0 },
			{ EBPF_OP_ADD64_IMM, 1, 0, 0, 1 }, // slot[0] += 1
			{ EBPF_OP_STXDW, 0, 1, 0, 0 },
			{ EBPF_OP_STW, 10, 0, -4, 1 }, // key = 1
			{ EBPF_OP_LDDW, 1, 1, 0, s.map_fd }, { 0, 0, 0, 0, 0 },
			{ EBPF_OP_MOV64_REG, 2, 10, 0, 0 },
			{ EBPF_OP_ADD64_IMM, 2, 0, 0, -4 },
			{ EBPF_OP_CALL, 0, 0, 0, 1 },
			{ EBPF_OP_JEQ_IMM, 0, 0, 3, 0 },
			{ EBPF_OP_LDXDW, 1, 0, 0, 0 },
			{ EBPF_OP_ADD64_REG, 1, 7, 0, 0 }, // slot[1] += arg1
			{ EBPF_OP_STXDW, 0, 1, 0, 0 },
			{ EBPF_OP_MOV64_IMM, 0, 0, 0, 0 },
			{ EBPF_OP_EXIT, 0, 0, 0, 0 },
		};
		s.uprobe_prog_fd = bpftime_progs_create(
			-1, uprobe_insns, std::size(uprobe_insns), "on_entry",
			static_cast<int>(bpf_prog_type::BPF_PROG_TYPE_KPROBE));
		require(s.uprobe_prog_fd >= 0, "uprobe program creation failed");

		// uretprobe: accumulate the return value (slot 2).
		const ebpf_inst uretprobe_insns[] = {
			{ EBPF_OP_LDXDW, 7, 1, RETURN_OFFSET, 0 }, // r7 = retval
			{ EBPF_OP_STW, 10, 0, -4, 2 }, // key = 2
			{ EBPF_OP_LDDW, 1, 1, 0, s.map_fd }, { 0, 0, 0, 0, 0 },
			{ EBPF_OP_MOV64_REG, 2, 10, 0, 0 },
			{ EBPF_OP_ADD64_IMM, 2, 0, 0, -4 },
			{ EBPF_OP_CALL, 0, 0, 0, 1 },
			{ EBPF_OP_JEQ_IMM, 0, 0, 3, 0 },
			{ EBPF_OP_LDXDW, 1, 0, 0, 0 },
			{ EBPF_OP_ADD64_REG, 1, 7, 0, 0 },
			{ EBPF_OP_STXDW, 0, 1, 0, 0 },
			{ EBPF_OP_MOV64_IMM, 0, 0, 0, 0 },
			{ EBPF_OP_EXIT, 0, 0, 0, 0 },
		};
		s.uret_prog_fd = bpftime_progs_create(
			-1, uretprobe_insns, std::size(uretprobe_insns),
			"on_return",
			static_cast<int>(bpf_prog_type::BPF_PROG_TYPE_KPROBE));
		require(s.uret_prog_fd >= 0, "uretprobe program creation failed");

		// Resolve the target module path and symbol offset portably.
		Dl_info info{};
		require(dladdr(reinterpret_cast<const void *>(
				 &bpftime_macos_demo_compute),
			     &info) != 0 &&
				info.dli_fname != nullptr && info.dli_fbase,
			"dladdr failed for target symbol");
		const uint64_t symbol_offset =
			reinterpret_cast<uintptr_t>(&bpftime_macos_demo_compute) -
			reinterpret_cast<uintptr_t>(info.dli_fbase);
		std::cout << "target module=" << info.dli_fname
			  << " symbol_offset=0x" << std::hex << symbol_offset
			  << std::dec << " mode=" << mode << "\n";

		s.uprobe_event_fd = bpftime_uprobe_create(
			-1, getpid(), info.dli_fname, symbol_offset, false, 0);
		s.uret_event_fd = bpftime_uprobe_create(
			-1, getpid(), info.dli_fname, symbol_offset, true, 0);
		require(s.uprobe_event_fd >= 0 && s.uret_event_fd >= 0,
			"bpftime_uprobe_create failed");

		bpf_link_create_args link_args{};
		link_args.prog_fd = s.uprobe_prog_fd;
		link_args.target_fd = s.uprobe_event_fd;
		link_args.attach_type = BPFTIME_BPF_PERF_EVENT_ATTACH_TYPE;
		s.uprobe_link_fd = bpftime_link_create(-1, &link_args);
		link_args.prog_fd = s.uret_prog_fd;
		link_args.target_fd = s.uret_event_fd;
		s.uret_link_fd = bpftime_link_create(-1, &link_args);
		require(s.uprobe_link_fd >= 0 && s.uret_link_fd >= 0,
			"bpftime_link_create failed");

		s.ctx.emplace();
		s.ctx->register_attach_impl(
			{ ATTACH_UPROBE, ATTACH_URETPROBE,
			  ATTACH_UPROBE_OVERRIDE, ATTACH_UREPLACE },
			std::make_unique<frida_attach_impl>(false),
			[](const std::string_view &spec, int &err)
				-> std::unique_ptr<attach_private_data> {
				auto priv =
					std::make_unique<frida_attach_private_data>();
				if (int e = priv->initialize_from_string(spec);
				    e < 0) {
					err = e;
					return nullptr;
				}
				return priv;
			});
		require(s.ctx->init_attach_ctx_from_handlers(config) == 0,
			"init_attach_ctx_from_handlers failed");

		uint64_t expected_arg_sum = 0, expected_ret_sum = 0;
		for (uint64_t i = 1; i <= iterations; ++i) {
			const uint64_t ret = bpftime_macos_demo_compute(i, 3);
			require(ret == i * 10 + 3, "target return value corrupted");
			expected_arg_sum += i;
			expected_ret_sum += ret;
		}

		auto read_slot = [&](uint32_t key) -> uint64_t {
			const void *ptr = bpftime_map_lookup_elem(s.map_fd, &key);
			require(ptr != nullptr, "map slot lookup failed");
			uint64_t value = 0;
			std::memcpy(&value, ptr, sizeof(value));
			return value;
		};
		require(read_slot(0) == iterations, "uprobe call count mismatch");
		require(read_slot(1) == expected_arg_sum,
			"uprobe argument sum mismatch");
		require(read_slot(2) == expected_ret_sum,
			"uretprobe return sum mismatch");
		std::cout << "calls=" << iterations
			  << " arg_sum=" << expected_arg_sum
			  << " ret_sum=" << expected_ret_sum << "\n";

		// Detach dynamically and prove the probes stop firing.
		require(s.ctx->destroy_instantiated_attach_link(
				s.uprobe_link_fd) == 0,
			"uprobe detach failed");
		require(s.ctx->destroy_instantiated_attach_link(
				s.uret_link_fd) == 0,
			"uretprobe detach failed");
		for (uint64_t i = 1; i <= 20; ++i)
			require(bpftime_macos_demo_compute(i, 3) == i * 10 + 3,
				"post-detach target call failed");
		require(read_slot(0) == iterations &&
				read_slot(1) == expected_arg_sum &&
				read_slot(2) == expected_ret_sum,
			"probe still fired after detach");

		s.close(); // deterministic cleanup before the leak checks
		require(open_fd_count() == fds_before, "file descriptors leaked");
		require(!shm_residue_exists(name), "shared memory leaked");
		std::cout << "macos_uprobe_demo " << mode
			  << ": PASSED (attach, exact maps, detach, cleanup)\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "FAILED: " << error.what() << "\n";
		return 1;
	}
}
