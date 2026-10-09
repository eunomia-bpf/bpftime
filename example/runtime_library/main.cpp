// SPDX-License-Identifier: MIT
#include "bpf_attach_ctx.hpp"
#include "bpftime_shm.hpp"
#include "simple_attach_impl.hpp"
#include <boost/interprocess/exceptions.hpp>
#include <cstdlib>
#include <charconv>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>

using namespace bpftime;

static void require(bool ok, const char *message)
{
	if (!ok)
		throw std::runtime_error(message);
}

// This executable owns the entire runtime session, including error paths.
struct session {
	bool owns_shm = false;
	int map_fd = -1, prog_fd = -1, event_fd = -1, link_fd = -1;
	std::optional<bpf_attach_ctx> ctx;

	~session()
	{
		if (ctx) {
			ctx->destroy_all_attach_links(); // Reset alone does not detach.
			ctx->reset_instantiated_state();
			ctx.reset(); // Release VMs and attach implementations before SHM.
		}
		for (int fd : { link_fd, event_fd, prog_fd, map_fd }) {
			if (fd >= 0) {
				bpftime_close(fd); // Remove the handler (may cascade).
				::close(fd); // These APIs allocated real /dev/null fds.
			}
		}
		if (owns_shm) {
			bpftime_destroy_global_shm(); // Unmap this process.
			bpftime_remove_global_shm(); // Unlink only our unique name.
		}
	}
};

static uint64_t read_counter(int fd)
{
	const uint32_t key = 0;
	const void *value = bpftime_map_lookup_elem(fd, &key);
	require(value != nullptr, "counter lookup failed");
	uint64_t count;
	std::memcpy(&count, value, sizeof(count));
	return count; // The returned map pointer must not outlive the map.
}

static uint64_t run_lifecycle(uint32_t events, bool jit)
{
	session s;
	// Never open or remove an existing session, even on a name collision.
	s.owns_shm = true; // Also clean up if registry allocation throws.
	try {
		bpftime_initialize_global_shm(shm_open_type::SHM_CREATE_ONLY);
	} catch (const boost::interprocess::interprocess_exception &error) {
		if (error.get_error_code() == boost::interprocess::already_exists_error)
			s.owns_shm = false; // An existing segment belongs to someone else.
		throw;
	}
	runtime_config config;
	config.set_vm_name("ubpf");
	config.jit_enabled = jit;
	config.enable_kernel_helper_group = false;
	config.enable_ufunc_helper_group = false;
	config.enable_shm_maps_helper_group = true;
	bpftime_set_runtime_config(runtime_config(config));

	// 1. Create and initialize a one-entry userspace counter map.
	bpf_map_attr attr {};
	attr.type = static_cast<uint32_t>(bpf_map_type::BPF_MAP_TYPE_ARRAY);
	attr.key_size = sizeof(uint32_t);
	attr.value_size = sizeof(uint64_t);
	attr.max_ents = 1;
	s.map_fd = bpftime_maps_create(-1, "counter", attr);
	require(s.map_fd >= 0, "map creation failed");
	const uint32_t key = 0;
	const uint64_t zero = 0;
	require(bpftime_map_update_elem(s.map_fd, &key, &zero, 0) == 0,
		"counter initialization failed");

	// 2. Readable eBPF: value = map_lookup(counter, &key); (*value)++.
	// LDDW with src_reg=1 tells the VM to resolve the map fd at load time.
	const ebpf_inst insns[] = {
		{ EBPF_OP_STW,       10, 0, -4, 0 }, // stack key = 0
		{ EBPF_OP_LDDW,       1, 1,  0, s.map_fd },
		{ 0,                 0, 0,  0, 0 }, // second half of LDDW
		{ EBPF_OP_MOV64_REG,  2, 10, 0, 0 }, // r2 = stack pointer
		{ EBPF_OP_ADD64_IMM,  2, 0,  0, -4 }, // r2 = &key
		{ EBPF_OP_CALL,       0, 0,  0, 1 }, // bpf_map_lookup_elem
		{ EBPF_OP_JEQ_IMM,    0, 0,  5, 0 }, // null -> failure
		{ EBPF_OP_LDXDW,      1, 0,  0, 0 }, // r1 = *value
		{ EBPF_OP_ADD64_IMM,  1, 0,  0, 1 }, // r1++
		{ EBPF_OP_STXDW,      0, 1,  0, 0 }, // *value = r1
		{ EBPF_OP_MOV64_IMM,  0, 0,  0, 0 }, // return success
		{ EBPF_OP_EXIT,       0, 0,  0, 0 },
		{ EBPF_OP_MOV64_IMM,  0, 0,  0, -1 }, // return failure
		{ EBPF_OP_EXIT,       0, 0,  0, 0 },
	};
	// Program type 0: a custom userspace event, with no kernel context ABI.
	s.prog_fd = bpftime_progs_create(-1, insns, std::size(insns),
					"count_event", 0);
	require(s.prog_fd >= 0, "program handler creation failed");

	// 3. Register a cooperative event source and create its handler/link.
	constexpr int event_type = 2001;
	s.ctx.emplace();
	// The interpreter checks accesses against the declared context/stack.
	// Expose this map's single value as context so its helper-returned pointer
	// is in bounds. The program itself does not read the context register.
	void *counter_memory = bpftime_get_array_map_raw_data(s.map_fd);
	require(counter_memory != nullptr, "array value memory unavailable");
	auto trigger = simple_attach::add_simple_attach_impl_to_attach_ctx(
		event_type,
		[counter_memory](const std::string &, const std::string &,
				 const attach::ebpf_run_callback &run) {
			uint64_t result = 0;
			int err = run(counter_memory, sizeof(uint64_t), &result);
			return err < 0 ? err : (result == 0 ? 0 : -1);
		},
		*s.ctx);
	require(trigger("before attach") == 1, "unattached event executed");
	s.event_fd = bpftime_add_custom_perf_event(event_type, "counter");
	require(s.event_fd >= 0, "event creation failed");
	bpf_link_create_args link {};
	link.prog_fd = s.prog_fd;
	link.target_fd = s.event_fd;
	link.attach_type = BPFTIME_BPF_PERF_EVENT_ATTACH_TYPE;
	s.link_fd = bpftime_link_create(-1, &link);
	require(s.link_fd >= 0, "link creation failed");

	// 4. Instantiate the VM, register map helpers, load and attach.
	require(s.ctx->init_attach_ctx_from_handlers(config) == 0,
		"attach context initialization failed");
	auto impl = s.ctx->get_attach_impl_by_attach_type(event_type);
	auto *source = impl ? dynamic_cast<simple_attach::simple_attach_impl *>(
				      *impl) : nullptr;
	// Initialization can return zero after an individual handler failure.
	require(source && source->is_attached(), "event was not attached");

	// 5. Trigger real BPF execution, then read the map.
	require(read_counter(s.map_fd) == 0, "new counter was not zero");
	for (uint32_t i = 0; i < events; ++i)
		require(trigger("event") == 0, "BPF execution failed");
	uint64_t count = read_counter(s.map_fd);
	require(count == events, "event count mismatch");

	// 6. Detach before releasing programs/maps and prove it stops execution.
	require(s.ctx->destroy_instantiated_attach_link(s.link_fd) == 0,
		"detach failed");
	require(!source->is_attached(), "event remained attached");
	require(trigger("after detach") == 1, "detached event executed");
	require(read_counter(s.map_fd) == count, "counter changed after detach");
	return count; // session releases context, handlers, fds and SHM.
}

static uint32_t number(std::string_view text)
{
	uint32_t value = 0;
	auto [end, err] = std::from_chars(text.data(), text.data() + text.size(),
					value);
	require(err == std::errc {} && end == text.data() + text.size(),
		"expected an unsigned integer");
	return value;
}

static size_t open_fd_count()
{
	auto entries = std::filesystem::directory_iterator("/proc/self/fd");
	return std::distance(begin(entries), end(entries));
}

int main(int argc, char **argv)
{
	try {
		require(argc <= 4, "usage: runtime_library [events [cycles [interpreter|jit]]]");
		const uint32_t events = argc > 1 ? number(argv[1]) : 5;
		const uint32_t cycles = argc > 2 ? number(argv[2]) : 1;
		const std::string mode = argc > 3 ? argv[3] : "interpreter";
		require(cycles > 0 && (mode == "interpreter" || mode == "jit"),
			"cycles must be positive; mode must be interpreter or jit");
		spdlog::set_level(spdlog::level::err);
		const std::string name = "bpftime-runtime-library-" +
			std::to_string(getpid()) + "-" +
			std::to_string(std::chrono::steady_clock::now()
					       .time_since_epoch().count());
		require(setenv("BPFTIME_GLOBAL_SHM_NAME", name.c_str(), 1) == 0,
			"setting private SHM name failed");
		std::cout << "shm=" << name << '\n';
		const size_t fds = open_fd_count();
		for (uint32_t i = 0; i < cycles; ++i) {
			uint64_t count = run_lifecycle(events, mode == "jit");
			require(open_fd_count() == fds, "file descriptors leaked");
			require(!std::filesystem::exists("/dev/shm/" + name),
				"shared memory leaked");
			std::cout << "cycle=" << i + 1 << " events=" << events
				  << " count=" << count << " detached_count="
				  << count << '\n';
		}
		std::cout << "cleanup=ok\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
