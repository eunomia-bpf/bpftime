#include "catch2/catch_message.hpp"
#include "../common_def.hpp"
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <bpf_map/userspace/per_cpu_hash_map.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <cstdint>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <bpf_map/map_common_def.hpp>
#include "handler/map_handler.hpp"
#include "catch2/internal/catch_run_context.hpp"
#include <algorithm>
#include <atomic>
#include <random>
#include <thread>

using namespace boost::interprocess;
using namespace bpftime;

static const char *SHM_NAME = "BPFTIME_PER_CPU_HASH_SHM";

TEST_CASE("Test basic operations of hash map")
{
	shm_remove remover(SHM_NAME);
	managed_shared_memory mem(create_only, SHM_NAME, 20 << 20);
	uint32_t ncpu = sysconf(_SC_NPROCESSORS_ONLN);
	std::mt19937 gen;
	gen.seed(Catch::rngSeed());
	// Since this is a hash map, so we generate something randomly as keys
	std::uniform_int_distribution<uint32_t> rand(0, UINT32_MAX);
	std::vector<uint32_t> keys;
	for (int i = 0; i < 100; i++) {
		auto x = rand(gen);
		while (std::find(keys.begin(), keys.end(), x) != keys.end())
			x = rand(gen);
		keys.push_back(x);
	}

	SECTION("Test writing from helpers, and read from userspace")
	{
		per_cpu_hash_map_impl map(mem, 4, 8, 1 << 20);
		for (uint32_t j = 0; j < ncpu; j++) {
			ensure_on_certain_cpu<void>(j, [&]() {
				for (uint32_t i = 0; i < 100; i++) {
					uint64_t val =
						(((uint64_t)keys[i]) << 32) | j;
					INFO("Set index " << keys[i] << " cpu "
							  << j << " to "
							  << val);
					REQUIRE(map.elem_update(&keys[i], &val,
								0) == 0);
				}
			});
		}
		for (uint32_t i = 0; i < 100; i++) {
			uint64_t *p =
				(uint64_t *)map.elem_lookup_userspace(&keys[i]);
			for (uint32_t j = 0; j < ncpu; j++) {
				REQUIRE(p[j] ==
					((((uint64_t)keys[i]) << 32) | j));
			}
		}
	}

	SECTION("Test writing from userspace, and reading & updating from helpers")
	{
		per_cpu_hash_map_impl map(mem, 4, 8, 1 << 20);
		std::vector<uint64_t> buf(ncpu);
		for (uint32_t j = 0; j < ncpu; j++) {
			buf[j] = j;
		}
		uint32_t dummy = 0xabcdef;
		REQUIRE(map.elem_update_userspace(&dummy, buf.data(), 0) == 0);
		// Update from helpers
		for (uint32_t i = 0; i < ncpu; i++) {
			ensure_on_certain_cpu<void>(i, [&]() {
				auto valuep =
					(uint64_t *)map.elem_lookup(&dummy);
				REQUIRE(valuep != nullptr);
				auto value = *valuep;
				value |= (value << 32);
				REQUIRE(map.elem_update(&dummy, &value, 0) ==
					0);
			});
		}
		// Read from helpers
		for (uint32_t i = ncpu - 1; ~i; i--) {
			ensure_on_certain_cpu<void>(i, [&]() {
				auto valuep =
					(uint64_t *)map.elem_lookup(&dummy);
				REQUIRE(valuep != nullptr);
				auto value = *valuep;
				uint64_t a = value >> 32;
				uint64_t b = (value << 32) >> 32;
				REQUIRE(a == b);
			});
		}
	}
}

TEST_CASE("Per-CPU hash helper flags preserve conditional update semantics",
	  "[maps][percpu_hash][update_flags]")
{
	shm_remove remover("BPFTIME_PER_CPU_HASH_FLAGS_SHM");
	managed_shared_memory mem(create_only, "BPFTIME_PER_CPU_HASH_FLAGS_SHM",
				 20 << 20);
	per_cpu_hash_map_impl map(mem, sizeof(uint32_t), sizeof(uint64_t), 1);
	uint32_t key = 1, missing = 2;
	uint64_t replacement = 22;
	std::vector<uint64_t> values(map.getncpu(), 11);
	REQUIRE(map.elem_update_userspace(&key, values.data(), 0) == 0);
	// The common flag validator reserves the high bits for custom operations;
	// they must not bypass BPF_NOEXIST/BPF_EXIST in the low 32 bits.
	for (uint64_t custom : { 0ULL, 1ULL << 32 }) {
		errno = 0;
		REQUIRE(map.elem_update(&key, &replacement, custom | 1) == -1);
		REQUIRE(errno == EEXIST);
		auto *stored = static_cast<uint64_t *>(map.elem_lookup_userspace(&key));
		REQUIRE(stored != nullptr);
		REQUIRE(std::equal(values.begin(), values.end(), stored));
		errno = 0;
		REQUIRE(map.elem_update(&missing, &replacement, custom | 2) == -1);
		REQUIRE(errno == ENOENT);
	}
	errno = 0;
	REQUIRE(map.elem_update(&missing, &replacement, 0) == -1);
	REQUIRE(errno == E2BIG);
	REQUIRE(map.elem_delete(&key) == 0);
	REQUIRE(map.elem_lookup_userspace(&key) == nullptr);
	REQUIRE(map.elem_lookup(&key) == nullptr);
	errno = 0;
	REQUIRE(map.elem_delete(&key) == -1);
	REQUIRE(errno == ENOENT);
	REQUIRE(map.elem_update(&missing, &replacement, 1) == 0);
}

TEST_CASE("Per-CPU hash map handler serializes concurrent helper and userspace access",
	  "[maps][percpu_hash][concurrency]")
{
	static constexpr const char *CONC_SHM = "BPFTIME_PER_CPU_HASH_CONC_SHM";
	shm_remove remover(CONC_SHM);
	managed_shared_memory segment(create_only, CONC_SHM, 4 << 20);
	bpf_map_handler map_hd(1, (int)bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH,
			       sizeof(uint32_t), sizeof(uint64_t), 128, 0,
			       "pcpu_conc", segment);
	REQUIRE(map_hd.map_init(segment) == 0);

	const long ncpu = std::max<long>(1, sysconf(_SC_NPROCESSORS_ONLN));
	std::atomic<int> failures{ 0 };
	std::vector<std::thread> workers;
	for (int t = 0; t < 4; t++) {
		workers.emplace_back([&map_hd, t, ncpu, &failures] {
			const uint64_t expected_user = uint64_t(t + 1);
			std::vector<uint64_t> user_vals(size_t(ncpu),
							expected_user);
			for (uint32_t iter = 0; iter < 300; iter++) {
				// Disjoint per-thread keys exercise concurrent
				// shared-table mutation and key_template reuse.
				uint32_t k = uint32_t(t * 32) + (iter % 16);
				uint64_t v = (uint64_t(t + 1) << 32) | iter;
				if ((iter & 1U) == 0U) {
					// Helper path: trylock, so EBUSY is a
					// counted skip, never a failure.
					long rc = map_hd.map_update_elem(
						&k, &v, 0, false);
					if (rc != 0) {
						if (rc != -EBUSY)
							failures.fetch_add(1);
						continue;
					}
					errno = EACCES;
					const void *helper_val =
						map_hd.map_lookup_elem(&k,
								       false);
					if (errno != EACCES)
						failures.fetch_add(1);
					(void)helper_val;
					// The fd-based syscall path blocks and
					// must always observe the stored slot.
					const auto *slots = static_cast<
						const uint64_t *>(
						map_hd.map_lookup_elem(&k,
								       true));
					if (slots == nullptr ||
					    std::find(slots, slots + ncpu, v) ==
						    slots + ncpu)
						failures.fetch_add(1);
				} else {
					// Syscall path keeps the blocking
					// lock and must not be skipped.
					if (map_hd.map_update_elem(
						    &k, user_vals.data(), 0,
						    true) != 0)
						failures.fetch_add(1);
					errno = EACCES;
					const auto *hval = static_cast<
						const uint64_t *>(
						map_hd.map_lookup_elem(&k,
								       false));
					if (errno != EACCES ||
					    (hval != nullptr && *hval != expected_user)) {
						failures.fetch_add(1);
					}
					const auto *slots = static_cast<
						const uint64_t *>(
						map_hd.map_lookup_elem(&k,
								       true));
					if (slots == nullptr ||
					    slots[0] != expected_user ||
					    slots[ncpu - 1] != expected_user)
						failures.fetch_add(1);
				}
				uint32_t next = 0;
				(void)map_hd.bpf_map_get_next_key(nullptr,
								  &next, true);
				if ((iter & 3U) == 3U &&
				    map_hd.map_delete_elem(&k, true) != 0)
					failures.fetch_add(1);
			}
		});
	}
	for (auto &w : workers)
		w.join();
	REQUIRE(failures.load() == 0);
	map_hd.map_free(segment);
}

TEST_CASE("Per-CPU hash helper trylock skips and counts held-lock events",
	  "[maps][percpu_hash][concurrency]")
{
	static constexpr const char *TRYLOCK_SHM =
		"BPFTIME_PER_CPU_HASH_TRYLOCK_SHM";
	shm_remove remover(TRYLOCK_SHM);
	managed_shared_memory segment(create_only, TRYLOCK_SHM, 4 << 20);
	bpf_map_handler map_hd(2, (int)bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH,
			       sizeof(uint32_t), sizeof(uint64_t), 128, 0,
			       "pcpu_trylock", segment);
	REQUIRE(map_hd.map_init(segment) == 0);

	const long ncpu = std::max<long>(1, sysconf(_SC_NPROCESSORS_ONLN));
	uint32_t key = 7;
	std::vector<uint64_t> user_vals(size_t(ncpu), 1);
	REQUIRE(map_hd.map_update_elem(&key, user_vals.data(), 0, true) == 0);

	const uint64_t skips_before = get_percpu_map_trylock_skip_count();
	// Hold the mutex in this thread: a blocking helper would hang here.
	REQUIRE(map_hd.get_map_lock().lock("trylock-test"));
	errno = EACCES;
	uint64_t helper_val = 99;
	const long update_result =
		map_hd.map_update_elem(&key, &helper_val, 0, false);
	const void *lookup_result = map_hd.map_lookup_elem(&key, false);
	const int helper_errno = errno;
	map_hd.get_map_lock().unlock();
	REQUIRE(update_result == -EBUSY);
	REQUIRE(lookup_result == nullptr);
	REQUIRE(helper_errno == EACCES);
	REQUIRE(get_percpu_map_trylock_skip_count() == skips_before + 2);

	REQUIRE(map_hd.map_update_elem(&key, &helper_val, 0, false) == 0);
	REQUIRE(map_hd.map_lookup_elem(&key, false) != nullptr);
	REQUIRE(errno == EACCES);
	map_hd.map_free(segment);
}
