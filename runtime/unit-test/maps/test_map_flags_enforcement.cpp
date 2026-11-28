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
#include <cstring>
#include <memory>
#include <string>
#include <unistd.h>
#include <variant>
#include <vector>

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

TEST_CASE("bpf_map_handler enforces BPF_F_RDONLY/BPF_F_WRONLY across keyed map backends",
	  "[map_flags]")
{
	const std::string shm_name =
		"_MAP_HANDLER_KEYED_FLAGS_TEST_" + std::to_string(getpid());
	::shm_remove remover(shm_name.c_str());
	managed_shared_memory segment(create_only, shm_name.c_str(), 16 << 20);
	auto *manager = segment.construct<handler_manager>(
		"_keyed_flags_handler_manager")(segment, MIN_MAX_FD_COUNT);

	int next_fd = 10;
	auto make_map = [&](bpftime::bpf_map_type mtype, uint32_t ksz,
			    uint32_t vsz, uint32_t max_ents,
			    uint64_t mflags) -> const bpf_map_handler & {
		const int fd = next_fd++;
		REQUIRE(manager->set_handler(
				fd,
				bpf_map_handler(fd, (int)mtype, ksz, vsz,
						max_ents, mflags, "flag_map",
						segment),
				segment) == fd);
		return std::get<bpf_map_handler>((*manager)[fd]);
	};

	auto check_keyed_rw = [&](const bpf_map_handler &ro,
				  const bpf_map_handler &wo, const void *key,
				  void *next_key, const void *h_val,
				  const void *s_val, size_t lad_sz,
				  bool has_lad, bool allow_helper_write,
				  bool allow_delete) {
		std::vector<uint8_t> lad_buf(lad_sz ? lad_sz : 1, 0);

		if (allow_helper_write) {
			REQUIRE(ro.map_update_elem(key, h_val, BPF_ANY, false) ==
				0);
			REQUIRE(ro.map_lookup_elem(key, false) != nullptr);
		}
		errno = 0;
		REQUIRE(ro.map_update_elem(key, s_val, BPF_ANY, true) == -1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(ro.map_delete_elem(key, true) == -1);
		REQUIRE(errno == EPERM);
		if (has_lad) {
			errno = 0;
			REQUIRE(ro.map_lookup_and_delete_elem(
					key, lad_buf.data(), true) == -1);
			REQUIRE(errno == EPERM);
		}
		if (allow_helper_write) {
			REQUIRE(ro.map_lookup_elem(key, true) != nullptr);
			REQUIRE(ro.bpf_map_get_next_key(nullptr, next_key,
							true) == 0);
			if (has_lad)
				REQUIRE(ro.map_lookup_and_delete_elem(
						key, lad_buf.data(), false) ==
					0);
			else if (allow_delete)
				REQUIRE(ro.map_delete_elem(key, false) == 0);
		}

		REQUIRE(wo.map_update_elem(key, s_val, BPF_ANY, true) == 0);
		errno = 0;
		REQUIRE(wo.map_lookup_elem(key, true) == nullptr);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(wo.bpf_map_get_next_key(nullptr, next_key, true) == -1);
		REQUIRE(errno == EPERM);
		if (has_lad) {
			errno = 0;
			REQUIRE(wo.map_lookup_and_delete_elem(
					key, lad_buf.data(), true) == -1);
			REQUIRE(errno == EPERM);
		}
		REQUIRE(wo.map_lookup_elem(key, false) != nullptr);
		REQUIRE(wo.bpf_map_get_next_key(nullptr, next_key, false) == 0);
		if (allow_delete)
			REQUIRE(wo.map_delete_elem(key, true) == 0);
	};

	// Default fixed-size hash map
	{
		uint32_t k = 5, nk = 0;
		uint64_t v = 0x11223344ULL;
		const auto &ro = make_map(bpftime::bpf_map_type::BPF_MAP_TYPE_HASH,
					  sizeof(k), sizeof(v), 16,
					  BPF_F_RDONLY);
		const auto &wo = make_map(bpftime::bpf_map_type::BPF_MAP_TYPE_HASH,
					  sizeof(k), sizeof(v), 16,
					  BPF_F_WRONLY);
		check_keyed_rw(ro, wo, &k, &nk, &v, &v, sizeof(v), true, true,
			       true);
	}

	// Array map
	{
		uint32_t k = 1, nk = 0;
		uint64_t v = 0x55667788ULL;
		const auto &ro = make_map(bpftime::bpf_map_type::BPF_MAP_TYPE_ARRAY,
					  sizeof(k), sizeof(v), 4,
					  BPF_F_RDONLY);
		const auto &wo = make_map(bpftime::bpf_map_type::BPF_MAP_TYPE_ARRAY,
					  sizeof(k), sizeof(v), 4,
					  BPF_F_WRONLY);
		check_keyed_rw(ro, wo, &k, &nk, &v, &v, 0, false, true, false);
	}

	// Per-CPU array and Per-CPU hash maps
	for (auto mtype :
	     { bpftime::bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY,
	       bpftime::bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH }) {
		uint32_t k = 2, nk = 0;
		uint64_t hval = 0x99aabbccULL;
		const auto &ro =
			make_map(mtype, sizeof(k), sizeof(uint64_t), 8,
				 BPF_F_RDONLY);
		const auto &wo =
			make_map(mtype, sizeof(k), sizeof(uint64_t), 8,
				 BPF_F_WRONLY);
		const size_t sys_sz = ro.get_userspace_value_size();
		std::vector<uint8_t> sval(sys_sz, 0x2a);
		const bool is_hash =
			mtype == bpftime::bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH;
		check_keyed_rw(ro, wo, &k, &nk, &hval, sval.data(), sys_sz,
			       is_hash, true, is_hash);
	}

	// LPM trie
	{
		struct lpm_k {
			uint32_t prefixlen;
			uint32_t data;
		};
		lpm_k k{ 32, 0x0a000001U }, nk{ 0, 0 };
		uint64_t v = 0xdeadbeefULL;
		const auto &ro = make_map(
			bpftime::bpf_map_type::BPF_MAP_TYPE_LPM_TRIE,
			sizeof(lpm_k), sizeof(v), 8, BPF_F_RDONLY);
		const auto &wo = make_map(
			bpftime::bpf_map_type::BPF_MAP_TYPE_LPM_TRIE,
			sizeof(lpm_k), sizeof(v), 8, BPF_F_WRONLY);
		check_keyed_rw(ro, wo, &k, &nk, &v, &v, 0, false, true, true);
	}

	// Perf event array
	{
		uint32_t k = 0, nk = 0;
		int32_t fd_val = 7;
		const auto &ro = make_map(
			bpftime::bpf_map_type::BPF_MAP_TYPE_PERF_EVENT_ARRAY,
			sizeof(k), sizeof(fd_val), 4, BPF_F_RDONLY);
		const auto &wo = make_map(
			bpftime::bpf_map_type::BPF_MAP_TYPE_PERF_EVENT_ARRAY,
			sizeof(k), sizeof(fd_val), 4, BPF_F_WRONLY);
		check_keyed_rw(ro, wo, &k, &nk, &fd_val, &fd_val, 0, false, true,
			       true);
	}

	// Stack trace map (entries populated via fill_stack_trace)
	{
		constexpr size_t val_sz = 4 * sizeof(uint64_t);
		uint32_t nk = 0;
		uint64_t dummy_buf[4] = {};
		const auto &ro = make_map(
			bpftime::bpf_map_type::BPF_MAP_TYPE_STACK_TRACE,
			sizeof(uint32_t), val_sz, 8, BPF_F_RDONLY);
		const auto &wo = make_map(
			bpftime::bpf_map_type::BPF_MAP_TYPE_STACK_TRACE,
			sizeof(uint32_t), val_sz, 8, BPF_F_WRONLY);

		auto ro_impl = ro.try_get_stack_trace_map_impl();
		auto wo_impl = wo.try_get_stack_trace_map_impl();
		REQUIRE(ro_impl.has_value());
		REQUIRE(wo_impl.has_value());
		const int ro_id =
			(*ro_impl)->fill_stack_trace({ 0x1000, 0x2000 }, true,
						     true);
		const int wo_id =
			(*wo_impl)->fill_stack_trace({ 0x3000, 0x4000 }, true,
						     true);
		REQUIRE(ro_id >= 0);
		REQUIRE(wo_id >= 0);
		uint32_t ro_k = static_cast<uint32_t>(ro_id);
		uint32_t wo_k = static_cast<uint32_t>(wo_id);

		errno = 0;
		REQUIRE(ro.map_update_elem(&ro_k, dummy_buf, 0, true) == -1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(ro.map_delete_elem(&ro_k, true) == -1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(ro.map_lookup_and_delete_elem(&ro_k, dummy_buf, true) ==
			-1);
		REQUIRE(errno == EPERM);
		REQUIRE(ro.map_lookup_elem(&ro_k, true) != nullptr);

		errno = 0;
		REQUIRE(wo.map_lookup_elem(&wo_k, true) == nullptr);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(wo.bpf_map_get_next_key(nullptr, &nk, true) == -1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(wo.map_lookup_and_delete_elem(&wo_k, dummy_buf, true) ==
			-1);
		REQUIRE(errno == EPERM);
		REQUIRE(wo.map_lookup_elem(&wo_k, false) != nullptr);
		REQUIRE(wo.map_lookup_and_delete_elem(&wo_k, dummy_buf,
						      false) == 0);
	}

	// Map-in-map types (array of maps & hash of maps; update/delete are
	// syscall-only by design)
	for (auto mtype :
	     { bpftime::bpf_map_type::BPF_MAP_TYPE_ARRAY_OF_MAPS,
	       bpftime::bpf_map_type::BPF_MAP_TYPE_HASH_OF_MAPS }) {
		uint32_t k = 0, nk = 0, inner_id = 17;
		const auto &ro =
			make_map(mtype, sizeof(k), sizeof(uint32_t), 4,
				 BPF_F_RDONLY);
		const auto &wo =
			make_map(mtype, sizeof(k), sizeof(uint32_t), 4,
				 BPF_F_WRONLY);
		const bool is_hash =
			mtype == bpftime::bpf_map_type::BPF_MAP_TYPE_HASH_OF_MAPS;
		check_keyed_rw(ro, wo, &k, &nk, &inner_id, &inner_id, 0, false,
			       false, is_hash);
		// Syscall reads remain allowed on RDONLY: HASH_OF_MAPS reports
		// ENOENT while unpopulated ARRAY_OF_MAPS leaves errno == 0 for
		// syscall_context to translate to ENOENT.
		errno = 0;
		REQUIRE(ro.map_lookup_elem(&k, true) == nullptr);
		if (is_hash)
			REQUIRE(errno == ENOENT);
		else
			REQUIRE(errno != EPERM);
	}

#ifdef BPFTIME_BUILD_WITH_LIBBPF
	// Prog array
	{
		uint32_t k = 0, nk = 0;
		int32_t fake_prog_fd = 9999;
		const auto &ro = make_map(
			bpftime::bpf_map_type::BPF_MAP_TYPE_PROG_ARRAY,
			sizeof(k), sizeof(int32_t), 4, BPF_F_RDONLY);
		const auto &wo = make_map(
			bpftime::bpf_map_type::BPF_MAP_TYPE_PROG_ARRAY,
			sizeof(k), sizeof(int32_t), 4, BPF_F_WRONLY);

		errno = 0;
		REQUIRE(ro.map_update_elem(&k, &fake_prog_fd, 0, true) == -1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(ro.map_delete_elem(&k, true) == -1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(ro.map_lookup_elem(&k, true) == nullptr);
		REQUIRE(errno == ENOENT);
		REQUIRE(ro.bpf_map_get_next_key(nullptr, &nk, true) == 0);

		errno = 0;
		REQUIRE(wo.map_lookup_elem(&k, true) == nullptr);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(wo.bpf_map_get_next_key(nullptr, &nk, true) == -1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(wo.map_lookup_elem(&k, false) == nullptr);
		REQUIRE(errno == ENOENT);
		REQUIRE(wo.bpf_map_get_next_key(nullptr, &nk, false) == 0);
		REQUIRE(wo.map_delete_elem(&k, true) == 0);
	}
#endif

	for (int fd = 10; fd < next_fd; ++fd)
		manager->clear_id_at(fd, segment);
	segment.destroy_ptr(manager);
}

TEST_CASE("bpf_map_handler enforces BPF_F_RDONLY/BPF_F_WRONLY on queue, stack, and bloom filter",
	  "[map_flags]")
{
	const std::string shm_name =
		"_MAP_HANDLER_KEYLESS_FLAGS_TEST_" + std::to_string(getpid());
	::shm_remove remover(shm_name.c_str());
	managed_shared_memory segment(create_only, shm_name.c_str(), 8 << 20);
	auto *manager = segment.construct<handler_manager>(
		"_keyless_flags_handler_manager")(segment, MIN_MAX_FD_COUNT);

	int next_fd = 10;
	auto make_map = [&](bpftime::bpf_map_type mtype, uint32_t vsz,
			    uint32_t max_ents,
			    uint64_t mflags) -> const bpf_map_handler & {
		const int fd = next_fd++;
		REQUIRE(manager->set_handler(
				fd,
				bpf_map_handler(fd, (int)mtype, 0, vsz, max_ents,
						mflags, "keyless_flag_map",
						segment),
				segment) == fd);
		return std::get<bpf_map_handler>((*manager)[fd]);
	};

	for (auto mtype : { bpftime::bpf_map_type::BPF_MAP_TYPE_QUEUE,
			    bpftime::bpf_map_type::BPF_MAP_TYPE_STACK }) {
		const auto &ro =
			make_map(mtype, sizeof(uint64_t), 8, BPF_F_RDONLY);
		const auto &wo =
			make_map(mtype, sizeof(uint64_t), 8, BPF_F_WRONLY);
		uint64_t val = 0xabcdef0123456789ULL, out = 0;

		REQUIRE(ro.map_push_elem(&val, BPF_ANY, false) == 0);
		errno = 0;
		REQUIRE(ro.map_push_elem(&val, BPF_ANY, true) == -1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(ro.map_update_elem(nullptr, &val, BPF_ANY, true) == -1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(ro.map_delete_elem(nullptr, true) == -1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(ro.map_pop_elem(&out, true) == -1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(ro.map_lookup_and_delete_elem(nullptr, &out, true) ==
			-1);
		REQUIRE(errno == EPERM);

		REQUIRE(ro.map_peek_elem(&out, true) == 0);
		REQUIRE(out == val);
		REQUIRE(ro.map_lookup_elem(nullptr, true) != nullptr);
		REQUIRE(ro.map_pop_elem(&out, false) == 0);
		REQUIRE(out == val);

		REQUIRE(wo.map_push_elem(&val, BPF_ANY, true) == 0);
		errno = 0;
		REQUIRE(wo.map_peek_elem(&out, true) == -1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(wo.map_lookup_elem(nullptr, true) == nullptr);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(wo.map_pop_elem(&out, true) == -1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(wo.map_lookup_and_delete_elem(nullptr, &out, true) ==
			-1);
		REQUIRE(errno == EPERM);

		out = 0;
		REQUIRE(wo.map_peek_elem(&out, false) == 0);
		REQUIRE(out == val);
		out = 0;
		REQUIRE(wo.map_pop_elem(&out, false) == 0);
		REQUIRE(out == val);
	}

	// Bloom filter
	{
		const auto &ro = make_map(
			bpftime::bpf_map_type::BPF_MAP_TYPE_BLOOM_FILTER,
			sizeof(uint64_t), 16, BPF_F_RDONLY);
		const auto &wo = make_map(
			bpftime::bpf_map_type::BPF_MAP_TYPE_BLOOM_FILTER,
			sizeof(uint64_t), 16, BPF_F_WRONLY);
		uint64_t val = 0xfeedfacecafebeefULL;
		uint64_t probe = val;

		REQUIRE(ro.map_push_elem(&val, BPF_ANY, false) == 0);
		errno = 0;
		REQUIRE(ro.map_push_elem(&val, BPF_ANY, true) == -1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(ro.map_update_elem(nullptr, &val, BPF_ANY, true) == -1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(ro.map_delete_elem(nullptr, true) == -1);
		REQUIRE(errno == EPERM);
		REQUIRE(ro.map_peek_elem(&probe, true) == 0);
		REQUIRE(ro.map_peek_elem(&probe, false) == 0);
		// elem_lookup is unsupported on bloom filter, but allowed
		// read on RDONLY must reach EOPNOTSUPP instead of EPERM.
		errno = 0;
		REQUIRE(ro.map_lookup_elem(nullptr, true) == nullptr);
		REQUIRE(errno == EOPNOTSUPP);

		REQUIRE(wo.map_push_elem(&val, BPF_ANY, true) == 0);
		errno = 0;
		REQUIRE(wo.map_peek_elem(&probe, true) == -1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(wo.map_lookup_elem(nullptr, true) == nullptr);
		REQUIRE(errno == EPERM);
		REQUIRE(wo.map_peek_elem(&probe, false) == 0);
	}

	for (int fd = 10; fd < next_fd; ++fd)
		manager->clear_id_at(fd, segment);
	segment.destroy_ptr(manager);
}
