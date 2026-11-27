#include <bpf_map/userspace/map_in_maps.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <thread>
#include <vector>

using namespace boost::interprocess;
using namespace bpftime;

namespace
{
struct shm_cleanup {
	explicit shm_cleanup(const char *name) : name(name)
	{
		shared_memory_object::remove(name);
	}
	~shm_cleanup()
	{
		shared_memory_object::remove(name);
	}
	const char *name;
};
} // namespace

TEST_CASE("Array-of-maps rejects out-of-range lookup keys",
	  "[maps][map_in_map]")
{
	static const char *shm_name = "bpftime_map_in_map_bounds_test";
	shm_cleanup cleanup(shm_name);
	managed_shared_memory segment(create_only, shm_name, 65536);
	array_map_of_maps_impl map(segment, 2);

	uint32_t invalid_key = 2;
	errno = 0;
	REQUIRE(map.elem_lookup(&invalid_key) == nullptr);
	REQUIRE(errno == ENOENT);
}

TEST_CASE("Hash-of-maps stores inner map descriptors",
	  "[maps][map_in_map]")
{
	static const char *shm_name = "bpftime_hash_of_maps_test";
	shm_cleanup cleanup(shm_name);
	managed_shared_memory segment(create_only, shm_name, 65536);
	hash_map_of_maps_impl map(segment, sizeof(uint32_t), 2, 0);

	uint32_t key = 42;
	int inner_fd = 17;
	REQUIRE(map.elem_update(&key, &inner_fd, 0) == 0);
	REQUIRE(map.elem_lookup(&key) == reinterpret_cast<void *>(17));

	uint32_t next_key = 0;
	REQUIRE(map.map_get_next_key(nullptr, &next_key) == 0);
	REQUIRE(next_key == key);
	REQUIRE(map.elem_delete(&key) == 0);
	REQUIRE(map.elem_lookup(&key) == nullptr);
	REQUIRE(errno == ENOENT);
}

TEST_CASE("Array-of-maps syscall lookup returns null for empty slot",
	  "[maps][map_in_map]")
{
	static const char *shm_name = "bpftime_array_map_in_map_syscall_test";
	shm_cleanup cleanup(shm_name);
	managed_shared_memory segment(create_only, shm_name, 65536);
	array_map_of_maps_impl map(segment, 2);

	uint32_t key = 0;
	// Never-updated slot stores 0; the syscall path returns nullptr (the
	// syscall layer maps that to ENOENT) instead of a pointer to the zero
	// value that memcpy would copy as a bogus inner fd.
	REQUIRE(map.elem_lookup_userspace(&key) == nullptr);

	// Out-of-range key also returns nullptr through the userspace path.
	uint32_t invalid_key = 2;
	REQUIRE(map.elem_lookup_userspace(&invalid_key) == nullptr);

	int inner_fd = 17;
	REQUIRE(map.elem_update(&key, &inner_fd, 0) == 0);
	auto *stored =
		static_cast<int *>(map.elem_lookup_userspace(&key));
	REQUIRE(stored != nullptr);
	REQUIRE(*stored == 17);
}

TEST_CASE("Hash-of-maps syscall lookup rejects zero inner fd",
	  "[maps][map_in_map]")
{
	static const char *shm_name = "bpftime_hash_map_in_map_syscall_test";
	shm_cleanup cleanup(shm_name);
	managed_shared_memory segment(create_only, shm_name, 65536);
	hash_map_of_maps_impl map(segment, sizeof(uint32_t), 4, 0);

	uint32_t key = 7;
	// Missing key: underlying lookup returns nullptr.
	REQUIRE(map.elem_lookup_userspace(&key) == nullptr);

	// Explicitly stored zero must also be rejected (exercises the *stored==0
	// branch, distinct from the nullptr branch above).
	int zero_fd = 0;
	REQUIRE(map.elem_update(&key, &zero_fd, 0) == 0);
	REQUIRE(map.elem_lookup_userspace(&key) == nullptr);

	int inner_fd = 17;
	REQUIRE(map.elem_update(&key, &inner_fd, 0) == 0);
	auto *stored =
		static_cast<int *>(map.elem_lookup_userspace(&key));
	REQUIRE(stored != nullptr);
	REQUIRE(*stored == 17);
}

TEST_CASE("Hash-of-maps supports lock-free concurrent helper lookups",
	  "[maps][map_in_map][concurrent]")
{
	REQUIRE_FALSE(hash_map_of_maps_impl::should_lock);

	static const char *shm_z = "bpftime_hash_of_maps_zero_cap_test";
	shm_cleanup cleanup_z(shm_z);
	managed_shared_memory seg_z(create_only, shm_z, 65536);
	hash_map_of_maps_impl zero_map(seg_z, sizeof(uint32_t), 0, 0);
	uint32_t zk = 1;
	int zfd = 9;
	REQUIRE(zero_map.elem_lookup(&zk) == nullptr);
	REQUIRE(zero_map.elem_update(&zk, &zfd, 0) == -1);
	REQUIRE(errno == E2BIG);

	static const char *shm_name = "bpftime_hash_of_maps_concurrent_test";
	shm_cleanup cleanup(shm_name);
	managed_shared_memory segment(create_only, shm_name, 1024 * 1024);
	hash_map_of_maps_impl map(segment, sizeof(uint32_t), 64, 0);

	constexpr uint32_t steady_keys = 16;
	for (uint32_t k = 0; k < steady_keys; ++k) {
		int fd = static_cast<int>(100 + k);
		REQUIRE(map.elem_update(&k, &fd, 0) == 0);
	}

	std::atomic<bool> start{ false };
	std::atomic<bool> stop{ false };
	std::atomic<uint64_t> failures{ 0 };
	std::atomic<uint64_t> reads{ 0 };

	std::vector<std::thread> readers;
	for (int t = 0; t < 8; ++t) {
		readers.emplace_back([&]() {
			while (!start.load(std::memory_order_acquire))
				std::this_thread::yield();
			while (!stop.load(std::memory_order_acquire)) {
				for (uint32_t k = 0; k < steady_keys; ++k) {
					auto val = reinterpret_cast<uintptr_t>(
						map.elem_lookup(&k));
					if (val != 100 + k && val != 200 + k)
						failures.fetch_add(1);
					auto *sp = static_cast<int *>(
						map.elem_lookup_userspace(&k));
					if (sp == nullptr ||
					    (*sp != static_cast<int>(100 + k) &&
					     *sp != static_cast<int>(200 + k)))
						failures.fetch_add(1);
				}
				// Key toggled between 0 and nonzero: userspace
				// snapshot must never return a pointer to zero.
				uint32_t toggle_k = steady_keys;
				if (auto *tp = static_cast<int *>(
					    map.elem_lookup_userspace(&toggle_k))) {
					if (*tp == 0)
						failures.fetch_add(1);
				}
				reads.fetch_add(1, std::memory_order_relaxed);
			}
		});
	}

	start.store(true, std::memory_order_release);
	while (reads.load(std::memory_order_acquire) == 0)
		std::this_thread::yield();
	for (int iter = 0; iter < 500; ++iter) {
		for (uint32_t k = 0; k < steady_keys; ++k) {
			int fd = (iter % 2 == 0) ? static_cast<int>(200 + k) :
						   static_cast<int>(100 + k);
			if (map.elem_update(&k, &fd, 0) != 0)
				failures.fetch_add(1);
		}
		uint32_t toggle_k = steady_keys;
		int toggle_fd = (iter % 2 == 0) ? 0 : 777;
		if (map.elem_update(&toggle_k, &toggle_fd, 0) != 0)
			failures.fetch_add(1);
		uint32_t churn_k = 32 + static_cast<uint32_t>(iter % 16);
		int churn_fd = 1000 + iter;
		if (map.elem_update(&churn_k, &churn_fd, 0) != 0 ||
		    map.elem_delete(&churn_k) != 0)
			failures.fetch_add(1);
	}
	stop.store(true, std::memory_order_release);

	for (auto &r : readers)
		r.join();
	REQUIRE(reads.load() >= 8);
	REQUIRE(failures.load() == 0);
}
