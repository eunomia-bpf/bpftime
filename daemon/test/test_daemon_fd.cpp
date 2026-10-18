/* SPDX-License-Identifier: MIT */
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <boost/scope_exit.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include "bpftime_shm.hpp"
#include "user/bpftime_driver.hpp"
#include "user/bpf_prog_insns.hpp"
#include "bpf_tracer.skel.h"

extern "C" {
int __real_bpf_prog_get_fd_by_id(__u32);
int __real_bpf_map_get_fd_by_id(__u32);
int __real_bpf_obj_get_info_by_fd(int, void *, __u32 *);
int __real_bpf_map__lookup_elem(const bpf_map *, const void *, size_t, void *,
				size_t, __u64);
int __real_close(int);
}

namespace
{
struct fd_test_state {
	int fd = -1;
	int closes = 0;
	int info_calls = 0;
	int fail_info_call = 0;
	bool program = false;
	bool fail_open = false;
	bool fail_lookup = false;
	~fd_test_state()
	{
		// Also release descriptors when testing the unfixed
		// implementation.
		if (fd >= 0 && fcntl(fd, F_GETFD) >= 0)
			__real_close(fd);
	}
	int open_fd()
	{
		if (fail_open) {
			errno = ENOENT;
			return -1;
		}
		return fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
	}
};
fd_test_state *active = nullptr;
} // namespace

extern "C" int __wrap_bpf_prog_get_fd_by_id(__u32 id)
{
	return active ? active->open_fd() : __real_bpf_prog_get_fd_by_id(id);
}

extern "C" int __wrap_bpf_map_get_fd_by_id(__u32 id)
{
	return active ? active->open_fd() : __real_bpf_map_get_fd_by_id(id);
}

extern "C" int __wrap_bpf_obj_get_info_by_fd(int fd, void *info, __u32 *len)
{
	if (!active)
		return __real_bpf_obj_get_info_by_fd(fd, info, len);
	// The program FD must remain usable during dependent-map enumeration.
	REQUIRE(fd == active->fd);
	REQUIRE(fcntl(fd, F_GETFD) >= 0);
	if (++active->info_calls == active->fail_info_call) {
		errno = EIO;
		return -1;
	}
	if (active->program) {
		auto &prog = *static_cast<bpf_prog_info *>(info);
		prog.type = BPF_PROG_TYPE_KPROBE;
		prog.nr_map_ids = 0;
	} else {
		auto &map = *static_cast<bpf_map_info *>(info);
		map.type = BPF_MAP_TYPE_ARRAY;
		map.key_size = map.value_size = sizeof(uint32_t);
		map.max_entries = 1;
	}
	return 0;
}

extern "C" int __wrap_bpf_map__lookup_elem(const bpf_map *map, const void *key,
					   size_t key_size, void *value,
					   size_t value_size, __u64 flags)
{
	if (!active)
		return __real_bpf_map__lookup_elem(map, key, key_size, value,
						   value_size, flags);
	if (active->fail_lookup) {
		errno = EIO;
		return -1;
	}
	REQUIRE(value_size == sizeof(bpf_insn_data));
	auto &captured = *static_cast<bpf_insn_data *>(value);
	captured = {};
	captured.code_len = 2;
	auto *insns = reinterpret_cast<ebpf_inst *>(captured.code);
	insns[0].code = EBPF_OP_MOV64_IMM;
	insns[1].code = EBPF_OP_EXIT;
	return 0;
}

extern "C" int __wrap_close(int fd)
{
	const int result = __real_close(fd);
	if (active && fd == active->fd) {
		active->closes++;
		// Successful close need not preserve errno.
		errno = EBUSY;
	}
	return result;
}

TEST_CASE("Daemon releases temporary kernel FDs on every return", "[daemon-fd]")
{
	const bool program = GENERATE(false, true);
	fd_test_state state;
	state.program = program;
	const char *old_name = getenv("BPFTIME_GLOBAL_SHM_NAME");
	const bool had_name = old_name != nullptr;
	const std::string previous_name = old_name ? old_name : "";
	const std::string shm_name =
		"bpftime_daemon_fd_" + std::to_string(getpid());
	REQUIRE(setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(), 1) == 0);
	BOOST_SCOPE_EXIT_ALL(=)
	{
		boost::interprocess::shared_memory_object::remove(
			shm_name.c_str());
		if (had_name)
			setenv("BPFTIME_GLOBAL_SHM_NAME", previous_name.c_str(),
			       1);
		else
			unsetenv("BPFTIME_GLOBAL_SHM_NAME");
	};
	bpf_tracer_bpf tracer = {};
	// The lookup wrapper supplies captured instructions without a kernel
	// map.
	tracer.maps.bpf_prog_insns_map = reinterpret_cast<bpf_map *>(&tracer);
	bpftime::bpftime_driver driver({}, &tracer);
	int id = 7;
	int expected = id;
	int expected_errno = 0;
	int expected_info_calls = program ? 2 : 1;

	SECTION("success")
	{
	}
	SECTION("object already exists")
	{
		if (program) {
			ebpf_inst insn = {};
			REQUIRE(bpftime_progs_create(id, &insn, 1, "existing",
						     BPF_PROG_TYPE_KPROBE) ==
				id);
		} else {
			bpftime::bpf_map_attr attr = {};
			attr.type = BPF_MAP_TYPE_ARRAY;
			attr.key_size = attr.value_size = sizeof(uint32_t);
			attr.max_ents = 1;
			REQUIRE(bpftime_maps_create(id, "existing", attr) ==
				id);
		}
		expected = 0;
		expected_info_calls = 1;
	}
	SECTION("FD acquisition fails")
	{
		state.fail_open = true;
		expected = -1;
		expected_errno = ENOENT;
		expected_info_calls = 0;
	}
	SECTION("metadata lookup fails")
	{
		state.fail_info_call = 1;
		expected = -1;
		expected_errno = EIO;
		expected_info_calls = 1;
	}
	SECTION("shared-memory insertion fails")
	{
		id = INT_MAX; // Beyond any supported handler table capacity.
		expected = -1;
		expected_info_calls = 1;
	}
	if (program) {
		SECTION("instruction relocation fails")
		{
			state.fail_lookup = true;
			expected = -1;
			expected_errno = EIO;
			expected_info_calls = 1;
		}
		SECTION("dependent-map enumeration fails")
		{
			state.fail_info_call = 2;
			expected = -1;
			expected_errno = EIO;
		}
	}
	active = &state;
	BOOST_SCOPE_EXIT_ALL(&)
	{
		active = nullptr;
	};
	const int result =
		program ? driver.bpftime_progs_create_server(id, 1234) :
			  driver.bpftime_maps_create_server(id);
	const int saved_errno = errno;
	active = nullptr;
	CHECK(result == expected);
	if (expected_errno)
		CHECK(saved_errno == expected_errno);
	CHECK(state.info_calls == expected_info_calls);
	CHECK(state.closes == (state.fail_open ? 0 : 1));
	if (!state.fail_open) {
		REQUIRE(state.fd >= 0);
		CHECK(fcntl(state.fd, F_GETFD) == -1);
		CHECK(errno == EBADF);
	}
}
