// SPDX-License-Identifier: MIT
//
// Copyright (c) 2026, eunomia-bpf org
// All rights reserved.
//
// Enforcement of BPF_F_RDONLY / BPF_F_WRONLY on the userspace hash map
// backends. The kernel applies these flags only to fd-based syscall access
// (the map fd carries O_RDONLY/O_WRONLY; see map_get_sys_perms() checks in
// kernel/bpf/syscall.c): a BPF_F_RDONLY map stays fully writable by its
// owning BPF program, whose access is governed by the separate
// BPF_F_RDONLY_PROG/BPF_F_WRONLY_PROG flags instead. The backend therefore
// rejects an operation only when from_syscall is true.
#include "catch2/catch_test_macros.hpp"
#include "linux/bpf.h"
#include "bpftime_handler.hpp"
#include "unit-test/common_def.hpp"
#include <bpf_map/userspace/var_hash_map.hpp>
#include <bpf_map/userspace/lru_var_hash_map.hpp>
#include "bpftime_shm.hpp"
#include <cerrno>
#include <memory>
#include <variant>

using namespace boost::interprocess;
using namespace bpftime;

namespace
{

struct shm_initializer {
	std::unique_ptr<::shm_remove> remover;
	std::unique_ptr<managed_shared_memory> mem;
	shm_initializer(const char *memory_name, size_t memory_size = 20 << 20)
	{
		remover = std::make_unique<::shm_remove>(memory_name);
		mem = std::make_unique<managed_shared_memory>(
		    create_only, memory_name, memory_size);
	}
};

// Syscall path (from_syscall == true): every forbidden operation must fail
// with EPERM, matching the kernel fd-mode checks.
template <typename MapImpl>
static void run_syscall_rdonly_checks(managed_shared_memory &mem)
{
	int key = 1, next_key = 0;
	int value = 1234;
	{
		MapImpl map(mem, sizeof(key), sizeof(value), 16, BPF_F_RDONLY);

		// The permission check happens before the
		// BPF_EXIST/NOEXIST/ENOENT/E2BIG logic, so all update flags
		// report EPERM rather than their usual errno.
		errno = 0;
		REQUIRE(map.elem_update(&key, &value, BPF_ANY, true) == -1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(map.elem_update(&key, &value, BPF_NOEXIST, true) ==
			-1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(map.elem_update(&key, &value, BPF_EXIST, true) == -1);
		REQUIRE(errno == EPERM);

		// Delete is a write as well; on an empty map the old code
		// returned ENOENT, the kernel requires EPERM first.
		errno = 0;
		REQUIRE(map.elem_delete(&key, true) == -1);
		REQUIRE(errno == EPERM);

		// Reads remain allowed: the map is simply empty.
		errno = 0;
		REQUIRE(map.elem_lookup(&key, true) == nullptr);
		REQUIRE(errno == ENOENT);
		errno = 0;
		REQUIRE(map.map_get_next_key(nullptr, &next_key, true) == -1);
		REQUIRE(errno == ENOENT);
	}
}

template <typename MapImpl>
static void run_syscall_wronly_checks(managed_shared_memory &mem)
{
	int key = 1, another_key = 2, next_key = 0;
	int value = 1234;
	{
		MapImpl map(mem, sizeof(key), sizeof(value), 16, BPF_F_WRONLY);

		// Writes are allowed on a write-only map.
		REQUIRE(map.elem_update(&key, &value, BPF_ANY, true) == 0);
		REQUIRE(map.elem_update(&another_key, &value, BPF_NOEXIST,
					true) == 0);

		// Reads must fail with EPERM.
		errno = 0;
		REQUIRE(map.elem_lookup(&key, true) == nullptr);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(map.map_get_next_key(nullptr, &next_key, true) == -1);
		REQUIRE(errno == EPERM);

		// Deletes are writes and stay allowed.
		REQUIRE(map.elem_delete(&another_key, true) == 0);
	}
}

// BPF helper path (from_syscall == false): creation flags must not restrict
// the owning program at all.
template <typename MapImpl>
static void run_helper_bypass_checks(managed_shared_memory &mem)
{
	int key = 1, next_key = 0;
	int value = 1234;
	{
		// A read-only map can still be mutated by its BPF program.
		MapImpl map(mem, sizeof(key), sizeof(value), 16, BPF_F_RDONLY);
		REQUIRE(map.elem_update(&key, &value, BPF_ANY, false) == 0);
		REQUIRE(map.elem_lookup(&key, false) != nullptr);
		REQUIRE(map.elem_delete(&key, false) == 0);
		errno = 0;
		REQUIRE(map.elem_lookup(&key, false) == nullptr);
		REQUIRE(errno == ENOENT);
	}
	{
		// A write-only map can still be read by its BPF program.
		MapImpl map(mem, sizeof(key), sizeof(value), 16, BPF_F_WRONLY);
		REQUIRE(map.elem_update(&key, &value, BPF_ANY, false) == 0);
		REQUIRE(map.elem_lookup(&key, false) != nullptr);
		REQUIRE(map.map_get_next_key(nullptr, &next_key, false) == 0);
		REQUIRE(next_key == key);
		REQUIRE(map.elem_delete(&key, false) == 0);
	}
}

template <typename MapImpl>
static void run_unflagged_checks(managed_shared_memory &mem)
{
	int key = 1, new_value = 5678, next_key = 0;
	int value = 1234;
	{
		MapImpl map(mem, sizeof(key), sizeof(value), 16, 0);

		REQUIRE(map.elem_update(&key, &value, BPF_NOEXIST, true) == 0);
		// Replacing an existing key must keep working.
		REQUIRE(map.elem_update(&key, &new_value, BPF_EXIST, true) ==
			0);

		void *ptr = map.elem_lookup(&key, true);
		REQUIRE(ptr != nullptr);
		REQUIRE(*static_cast<int *>(ptr) == new_value);

		REQUIRE(map.map_get_next_key(nullptr, &next_key, true) == 0);
		REQUIRE(next_key == key);

		REQUIRE(map.elem_delete(&key, true) == 0);
		errno = 0;
		REQUIRE(map.elem_lookup(&key, true) == nullptr);
		REQUIRE(errno == ENOENT);
	}
}

} // namespace

TEST_CASE("var-size hash map enforces BPF_F_RDONLY/BPF_F_WRONLY",
	  "[map_flags]")
{
	shm_initializer shm("_VAR_HASH_MAP_FLAGS_TEST");
	auto &mem = *shm.mem;
	run_syscall_rdonly_checks<var_size_hash_map_impl>(mem);
	run_syscall_wronly_checks<var_size_hash_map_impl>(mem);
	run_helper_bypass_checks<var_size_hash_map_impl>(mem);
	run_unflagged_checks<var_size_hash_map_impl>(mem);

	int key = 1, value = 1234, out = 0;
	// lookup_and_delete both reads a value and deletes the entry, so the
	// syscall path needs both permissions: either flag rejects it.
	{
		var_size_hash_map_impl rdonly(mem, sizeof(key), sizeof(value),
					      16, BPF_F_RDONLY);
		errno = 0;
		REQUIRE(rdonly.lookup_and_delete(&key, &out, true) == -1);
		REQUIRE(errno == EPERM);
	}
	{
		var_size_hash_map_impl wronly(mem, sizeof(key), sizeof(value),
					      16, BPF_F_WRONLY);
		REQUIRE(wronly.elem_update(&key, &value, BPF_ANY, true) == 0);
		errno = 0;
		REQUIRE(wronly.lookup_and_delete(&key, &out, true) == -1);
		REQUIRE(errno == EPERM);
		// The rejected operation must not have removed the entry.
		REQUIRE(wronly.elem_update(&key, &value, BPF_EXIST, true) ==
			0);
	}
	{
		// Unflagged syscall lookup_and_delete returns the value and
		// removes the entry; the helper path on a read-only map is
		// unrestricted as well.
		var_size_hash_map_impl plain(mem, sizeof(key), sizeof(value),
					     16, 0);
		REQUIRE(plain.elem_update(&key, &value, BPF_ANY, true) == 0);
		REQUIRE(plain.lookup_and_delete(&key, &out, true) == 0);
		REQUIRE(out == value);
		errno = 0;
		REQUIRE(plain.elem_lookup(&key, true) == nullptr);
		REQUIRE(errno == ENOENT);

		var_size_hash_map_impl rdonly(mem, sizeof(key), sizeof(value),
					      16, BPF_F_RDONLY);
		REQUIRE(rdonly.elem_update(&key, &value, BPF_ANY, false) == 0);
		REQUIRE(rdonly.lookup_and_delete(&key, &out, false) == 0);
		REQUIRE(out == value);
	}
}

TEST_CASE("LRU hash map enforces BPF_F_RDONLY/BPF_F_WRONLY", "[map_flags]")
{
	shm_initializer shm("_LRU_HASH_MAP_FLAGS_TEST");
	auto &mem = *shm.mem;
	run_syscall_rdonly_checks<lru_var_hash_map_impl>(mem);
	run_syscall_wronly_checks<lru_var_hash_map_impl>(mem);
	run_helper_bypass_checks<lru_var_hash_map_impl>(mem);
	run_unflagged_checks<lru_var_hash_map_impl>(mem);
}

// End-to-end check through bpf_map_handler dispatch (the path used by both
// BPF helpers, from_syscall=false, and the syscall wrappers,
// from_syscall=true) for BPF_MAP_TYPE_LRU_HASH.
TEST_CASE("LRU hash map flags distinguish helper and syscall access",
	  "[map_flags]")
{
	static const char *shm_name = "_LRU_HASH_MAP_HANDLER_FLAGS_TEST";
	::shm_remove remover(shm_name);
	managed_shared_memory segment(create_only, shm_name, 8 << 20);
	auto *manager = segment.construct<handler_manager>(
		"_lru_flags_handler_manager")(segment, MIN_MAX_FD_COUNT);

	constexpr int rdonly_fd = 1;
	constexpr int wronly_fd = 2;
	REQUIRE(manager->set_handler(
			rdonly_fd,
			bpf_map_handler(rdonly_fd, BPF_MAP_TYPE_LRU_HASH,
					sizeof(uint32_t), sizeof(uint64_t), 16,
					BPF_F_RDONLY, "lru_rdonly", segment),
			segment) == rdonly_fd);
	REQUIRE(manager->set_handler(
			wronly_fd,
			bpf_map_handler(wronly_fd, BPF_MAP_TYPE_LRU_HASH,
					sizeof(uint32_t), sizeof(uint64_t), 16,
					BPF_F_WRONLY, "lru_wronly", segment),
			segment) == wronly_fd);

	uint32_t key = 1, next_key = 0;
	uint64_t value = 1234;

	{
		auto &map =
			std::get<bpf_map_handler>((*manager)[rdonly_fd]);
		// The owning BPF program keeps full read/write access.
		REQUIRE(map.map_update_elem(&key, &value, BPF_ANY, false) ==
			0);
		REQUIRE(map.map_lookup_elem(&key, false) != nullptr);
		REQUIRE(map.map_delete_elem(&key, false) == 0);
		REQUIRE(map.map_update_elem(&key, &value, BPF_ANY, false) ==
			0);

		// Syscall writes to a read-only map fail with EPERM.
		errno = 0;
		REQUIRE(map.map_update_elem(&key, &value, BPF_ANY, true) ==
			-1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(map.map_delete_elem(&key, true) == -1);
		REQUIRE(errno == EPERM);
		// Syscall reads on a read-only map stay allowed.
		REQUIRE(map.map_lookup_elem(&key, true) != nullptr);
	}

	{
		auto &map =
			std::get<bpf_map_handler>((*manager)[wronly_fd]);
		// The owning BPF program keeps full read/write access.
		REQUIRE(map.map_update_elem(&key, &value, BPF_ANY, false) ==
			0);
		REQUIRE(map.map_lookup_elem(&key, false) != nullptr);
		REQUIRE(map.bpf_map_get_next_key(nullptr, &next_key, false) ==
			0);
		REQUIRE(next_key == key);

		// Syscall reads from a write-only map fail with EPERM.
		errno = 0;
		REQUIRE(map.map_lookup_elem(&key, true) == nullptr);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(map.bpf_map_get_next_key(nullptr, &next_key, true) ==
			-1);
		REQUIRE(errno == EPERM);
		// Syscall writes on a write-only map stay allowed.
		REQUIRE(map.map_update_elem(&key, &value, BPF_ANY, true) == 0);
	}

	manager->clear_id_at(rdonly_fd, segment);
	manager->clear_id_at(wronly_fd, segment);
	segment.destroy_ptr(manager);
}

#if __linux__
// End-to-end through the global C shim, the layer shared by the intercepted
// bpf() syscall (syscall_context, from_syscall=true via the plain wrappers)
// and the GPU BPF-program relay in bpf_attach_ctx_cuda.cpp
// (from_syscall=false via the _with_origin helpers). BPF_MAP_TYPE_LRU_HASH
// is the creation-flag-aware backend reachable through this dispatch in the
// default build.
TEST_CASE("map shim separates syscall and BPF-helper access origin",
	  "[map_flags]")
{
	::shm_remove remover("_MAP_FLAGS_SHIM_TEST_SHM");
	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);

	constexpr int lru_hash_type =
		(int)bpftime::bpf_map_type::BPF_MAP_TYPE_LRU_HASH;
	constexpr int rdonly_fd = 1201;
	constexpr int wronly_fd = 1202;
	REQUIRE(bpftime_maps_create(
			rdonly_fd, "lru_rdonly_shim",
			bpf_map_attr{
				.type = lru_hash_type,
				.key_size = sizeof(uint32_t),
				.value_size = sizeof(uint64_t),
				.max_ents = 16,
				.flags = BPF_F_RDONLY }) >= 0);
	REQUIRE(bpftime_maps_create(
			wronly_fd, "lru_wronly_shim",
			bpf_map_attr{
				.type = lru_hash_type,
				.key_size = sizeof(uint32_t),
				.value_size = sizeof(uint64_t),
				.max_ents = 16,
				.flags = BPF_F_WRONLY }) >= 0);

	uint32_t key = 7;
	uint64_t value = 0x1122334455667788ULL;

	// A read-only map stays fully writable by its BPF program.
	REQUIRE(bpftime_map_update_elem_with_origin(
			rdonly_fd, &key, &value, BPF_ANY, false) == 0);
	REQUIRE(bpftime_map_lookup_elem_with_origin(rdonly_fd, &key, false) !=
		nullptr);
	REQUIRE(bpftime_map_delete_elem_with_origin(rdonly_fd, &key, false) ==
		0);
	REQUIRE(bpftime_map_update_elem_with_origin(
			rdonly_fd, &key, &value, BPF_ANY, false) == 0);

	// The same writes through the syscall shim fail with EPERM.
	errno = 0;
	REQUIRE(bpftime_map_update_elem(rdonly_fd, &key, &value, BPF_ANY) ==
		-1);
	REQUIRE(errno == EPERM);
	errno = 0;
	REQUIRE(bpftime_map_delete_elem(rdonly_fd, &key) == -1);
	REQUIRE(errno == EPERM);
	// Reads remain allowed on a read-only map.
	REQUIRE(bpftime_map_lookup_elem(rdonly_fd, &key) != nullptr);

	// A write-only map stays readable by its BPF program.
	REQUIRE(bpftime_map_update_elem_with_origin(
			wronly_fd, &key, &value, BPF_ANY, false) == 0);
	REQUIRE(bpftime_map_lookup_elem_with_origin(wronly_fd, &key, false) !=
		nullptr);

	// Syscall reads from the write-only map fail with EPERM.
	errno = 0;
	REQUIRE(bpftime_map_lookup_elem(wronly_fd, &key) == nullptr);
	REQUIRE(errno == EPERM);
	// Syscall writes remain allowed on a write-only map.
	REQUIRE(bpftime_map_update_elem(wronly_fd, &key, &value, BPF_ANY) == 0);
	// The BPF program may delete as well.
	REQUIRE(bpftime_map_delete_elem_with_origin(wronly_fd, &key, false) ==
		0);

	bpftime_destroy_global_shm();
}
#endif
